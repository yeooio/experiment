"""Formal 128-trial hyperparameter search for the three final datasets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from hp_model import MODEL_PATH, build_search_model


HP_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = HP_ROOT / "inputs"
RESULT_ROOT = HP_ROOT / "results"
SEED = 42

SEARCH_SPACE = {
    "learning_rate": [0.001, 0.01],
    "depth": [1, 2, 4, 8],
    "representation_dim": [16, 32, 64, 128],
    "dense_hidden": [16, 32, 64, 128],
}

FIXED = {
    "seq_len": 5,
    "dropout": 0.01,
    "epochs": 500,
    "n_agents": 2,
    "n_heads": 4,
    "seed": SEED,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
}

DATASETS = {
    "cs2": {
        "features": ["CCCT_3p8_4p0", "IC_peak"],
        "cells": {"train": "CS2_36", "in_selection": "CS2_37", "pure_validation": "CS2_38"},
        "batch_size": 64,
        "weight_decay": 0.0003,
    },
    "cx2": {
        "features": ["CCCT_3p8_4p0", "CCCT_4p16_4p17"],
        "cells": {"train": "CX2_36", "in_selection": "CX2_37", "pure_validation": "CX2_38"},
        "batch_size": 128,
        "weight_decay": 0.0003,
    },
    "cx2_385_405": {
        "features": ["F1_CCCT"],
        "cells": {"train": "CX2_36", "in_selection": "CX2_37", "pure_validation": "CX2_38"},
        "feature_definition": "CCCT[3.85,4.05] single feature",
        "batch_size": 128,
        "weight_decay": 0.0003,
    },
    "oxford": {
        "features": ["CCCT_V_main"],
        "cells": {"train": "Cell1", "in_selection": "Cell2", "pure_validation": "Cell3"},
        "batch_size": 128,
        "weight_decay": 0.0003,
    },
}


class BatteryDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, seq_len: int):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.x) - self.seq_len

    def __getitem__(self, index: int):
        return self.x[index : index + self.seq_len], self.y[index + self.seq_len].unsqueeze(-1)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trial_grid():
    return list(
        itertools.product(
            SEARCH_SPACE["learning_rate"],
            SEARCH_SPACE["depth"],
            SEARCH_SPACE["representation_dim"],
            SEARCH_SPACE["dense_hidden"],
        )
    )


def input_files(dataset: str) -> dict[str, Path]:
    return {
        role: INPUT_ROOT / dataset / f"{role}.csv"
        for role in ("train", "in_selection", "pure_validation")
    }


def load_frames(dataset: str) -> dict[str, pd.DataFrame]:
    spec = DATASETS[dataset]
    columns = spec["features"] + ["SoH"]
    frames = {}
    for role, path in input_files(dataset).items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path).dropna(subset=columns).reset_index(drop=True)
        if len(frame) <= FIXED["seq_len"]:
            raise ValueError(f"{path} has only {len(frame)} valid rows")
        frames[role] = frame
    return frames


def build_loaders(dataset: str, seed: int):
    spec = DATASETS[dataset]
    frames = load_frames(dataset)
    scaler_x = StandardScaler().fit(frames["train"][spec["features"]].to_numpy(float))
    scaler_y = MinMaxScaler().fit(frames["train"][["SoH"]].to_numpy(float))
    loaders = {}
    for role, frame in frames.items():
        x = scaler_x.transform(frame[spec["features"]].to_numpy(float))
        y = scaler_y.transform(frame[["SoH"]].to_numpy(float)).ravel()
        shuffle = role == "train"
        loaders[role] = DataLoader(
            BatteryDataset(x, y, FIXED["seq_len"]),
            batch_size=spec["batch_size"],
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed) if shuffle else None,
            num_workers=0,
        )
    return loaders, scaler_y, frames


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.item()) * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, scaler_y, device) -> dict[str, float]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ys.append(y.numpy())
        ps.append(model(x.to(device)).cpu().numpy())
    y_true = scaler_y.inverse_transform(np.concatenate(ys).reshape(-1, 1)).ravel()
    y_pred = scaler_y.inverse_transform(np.concatenate(ps).reshape(-1, 1)).ravel()
    residual = y_true - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    mask = np.abs(y_true) > 1e-12
    return {
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "MAPE_pct": float(np.mean(np.abs(residual[mask] / y_true[mask])) * 100.0),
    }


def ranking_key(result: dict):
    metric = result["in_selection"]
    hp = result["hyperparameters"]
    return (
        -metric["R2"],
        metric["RMSE"],
        metric["MAE"],
        metric["MAPE_pct"],
        result["n_params"],
        hp["depth"],
    )


def write_summaries(dataset: str, out_dir: Path, results: list[dict]) -> dict:
    ordered = sorted(results, key=ranking_key)
    rows = []
    for result in ordered:
        row = {
            "trial_id": result["trial_id"],
            **result["hyperparameters"],
            "n_params": result["n_params"],
            "elapsed_s": result["elapsed_s"],
            "last_train_loss": result["last_train_loss"],
        }
        for role in ("train", "in_selection", "pure_validation"):
            for metric, value in result[role].items():
                row[f"{role}_{metric}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "all_trials.csv", index=False, encoding="utf-8-sig")
    best = ordered[0]
    (out_dir / "best.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    frozen = {
        "dataset": dataset,
        "selection_rule": "highest in_selection R2; pure_validation never participates",
        "source_trial": best["trial_id"],
        "hyperparameters": best["hyperparameters"],
        "in_selection": best["in_selection"],
        "pure_validation": best["pure_validation"],
    }
    (out_dir / "frozen_hp.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    return best


def run_dataset(dataset: str, device_name: str, limit: int | None, epochs_override: int | None, run_tag: str) -> None:
    spec = DATASETS[dataset]
    out_dir = RESULT_ROOT / run_tag / dataset
    trial_dir = out_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(8)

    grid = trial_grid()
    if limit is not None:
        grid = grid[:limit]
    epochs = epochs_override if epochs_override is not None else FIXED["epochs"]
    _, _, frames = build_loaders(dataset, SEED)
    files = input_files(dataset)
    manifest = {
        "created_at": now(),
        "dataset": dataset,
        "architecture": "Lite-PNAA+K5+K31+TunableFFN+AffineFreeLN+FlattenLinear",
        "model_source": str(MODEL_PATH),
        "search_model_source": str(Path(__file__).with_name("hp_model.py")),
        "device": str(device),
        "search_space": SEARCH_SPACE,
        "fixed": {**FIXED, "epochs": epochs, "batch_size": spec["batch_size"], "weight_decay": spec["weight_decay"]},
        "features": spec["features"],
        "cells": spec["cells"],
        "files": {role: str(path) for role, path in files.items()},
        "file_sha256": {role: sha256(path) for role, path in files.items()},
        "valid_rows": {role: len(frame) for role, frame in frames.items()},
        "selection_protocol": "train trains; in_selection selects HP; pure_validation reports only",
        "total_trials": len(grid),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    results = []
    for trial_id, (lr, depth, d_model, dense_hidden) in enumerate(grid, start=1):
        path = trial_dir / f"trial_{trial_id:03d}.json"
        if path.exists():
            try:
                results.append(json.loads(path.read_text(encoding="utf-8")))
                print(f"[{now()}][{dataset}] RESUME-SKIP trial={trial_id}/{len(grid)}", flush=True)
                continue
            except (json.JSONDecodeError, KeyError):
                print(f"[{now()}][{dataset}] INVALID trial={trial_id}; rerunning", flush=True)

        hp = {
            "learning_rate": lr,
            "depth": depth,
            "representation_dim": d_model,
            "dense_hidden": dense_hidden,
            **FIXED,
            "epochs": epochs,
            "batch_size": spec["batch_size"],
            "weight_decay": spec["weight_decay"],
        }
        print(
            f"[{now()}][{dataset}] START trial={trial_id}/{len(grid)} lr={lr} depth={depth} "
            f"d_model={d_model} dense_hidden={dense_hidden} device={device}",
            flush=True,
        )
        seed_everything(SEED)
        loaders, scaler_y, _ = build_loaders(dataset, SEED)
        model = build_search_model(
            input_dim=len(spec["features"]),
            d_model=d_model,
            dense_hidden=dense_hidden,
            depth=depth,
            n_agents=FIXED["n_agents"],
            n_heads=FIXED["n_heads"],
            dropout=FIXED["dropout"],
            seq_len=FIXED["seq_len"],
        ).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=spec["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=max(int(epochs * 1.2), 1))
        criterion = nn.MSELoss()
        started = time.time()
        losses = []
        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
            losses.append(loss)
            scheduler.step()
            if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
                print(
                    f"[{now()}][{dataset}] trial={trial_id}/{len(grid)} epoch={epoch}/{epochs} loss={loss:.10f}",
                    flush=True,
                )
        scores = {
            role: evaluate(model, loaders[role], scaler_y, device)
            for role in ("train", "in_selection", "pure_validation")
        }
        result = {
            "trial_id": trial_id,
            "dataset": dataset,
            "device": str(device),
            "hyperparameters": hp,
            "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "elapsed_s": round(time.time() - started, 3),
            "last_train_loss": losses[-1],
            "train_loss_history": losses,
            **scores,
        }
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(result)
        best = write_summaries(dataset, out_dir, results)
        print(
            f"[{now()}][{dataset}] DONE trial={trial_id}/{len(grid)} "
            f"in_R2={scores['in_selection']['R2']:.9f} pure_R2={scores['pure_validation']['R2']:.9f} "
            f"BEST=trial{best['trial_id']:03d}/in_R2={best['in_selection']['R2']:.9f} "
            f"elapsed={result['elapsed_s']:.1f}s",
            flush=True,
        )
        del model, loaders, scaler_y, optimizer, scheduler, criterion
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summaries(dataset, out_dir, results)
    print(f"[{now()}][{dataset}] COMPLETE trials={len(results)}/{len(grid)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs-override", type=int)
    parser.add_argument("--run-tag", default="formal_128x500")
    args = parser.parse_args()
    run_dataset(args.dataset, args.device, args.limit, args.epochs_override, args.run_tag)


if __name__ == "__main__":
    main()
