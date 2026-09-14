# 最新：5090三路并行搜索

本次更新仅提供5090任务，不包含4080运行包。

- [下载5090压缩包](SOH_5090_CS2_OXFORD_3LANES.zip)
- [5090运行说明](README_5090.md)
- 代码目录：`SOH_5090_CS2_OXFORD/`

固定种子1，搜索100和1000轮，每个组合只训练一次。三路分别为CS2其余四个模型（1664次）、Oxford全部模型（2432次）、CS2 Transformer（768次），共4864次且互不重复。

以下为保留的历史H100方案，不是本次5090启动方式。

## 历史：SOH hyperparameter search on one H100

This repository runs the complete four-dataset search for CS2, CX2, MIT/Severson, and Oxford. One GPU process is launched per dataset on the same H100.

The fixed protocol uses 500 epochs and seeds 1-10. It contains 48,640 uniquely identified training tasks. Model weights are trained only on the training cell, while hyperparameters are selected using the configuration cell. Completed tasks retain `result.json`; checkpoints are retained only for interrupted tasks and removed after successful completion.

Read [README_H100.md](README_H100.md) before running. On Windows, use `START_H100.cmd`. On Linux, prepare the documented Python/PyTorch environment and run `start_h100.sh`.

Do not edit `runner.py`, `source/`, `inputs/`, or `manifest.json` after a run begins. Their hashes define the experiment protocol.
