New search protocol: one fixed seed (1); epochs 100 and 1000; one run per complete combination.
Other search ranges: lr .001/.01; depth 1/2/4/8; width and FFN 16/32/64/128; attention heads 1/2/4.
All five models, all four datasets: 9728 tasks total.
Each machine owns 4864 disjoint tasks, assigned by DATASET, not parameter shards.
This 5090 package runs THREE parallel lanes: CS2 excluding Transformer; Oxford all models; CS2 Transformer only.
The separate 4080 package runs CX2 and MIT in TWO parallel lanes.
This prioritizes lower rental cost, not equal machine finish times. Do NOT run both identities on one machine.
Use this new SOH_DATASET_SPLIT_20260914 folder, not the previous half-grid package.

Keep old package/results unchanged. Stop the OLD training jobs on each machine before starting this new package.
Old 500-epoch checkpoints/results must NOT be copied here. No software is automatically installed.
Activate your working Python/PyTorch environment first. Run in this extracted directory:
  5090: python start.py 5090
  4080: python start.py 4080
On AutoDL you may use /root/miniconda3/bin/python instead of python.
Keep terminal open until checks complete and a background PID is printed.
On Windows use the Python executable from your existing working GPU environment.
start.py checks all five models, then launches THREE parallel GPU workers on the 5090.
Logs: cs2_other.log, oxford.log, cs2_transformer.log; supervisor log: 5090.log.
Logs: logs/5090.log or logs/4080.log. Results: results/5090 or results/4080.
Summary: python summary.py
After both machines finish, copy their separate results subfolders together then run summary.py.
Do not call a provisional best result the final selected configuration before its full grid completes.

Unchanged training rules: batch 128, window 5, dropout .01, AdamW weight decay .0003,
gradient clipping via archived training code, cosine scheduler T_max=600, no early stopping.
The scheduler remains the approved existing rule, including beyond epoch 600 for 1000-epoch trials.
100 and 1000 are separately initialized runs; final epoch configuration RMSE is used, not best-epoch selection.
CNN-Transformer pools only in its first encoder. Original source snapshots and data processing reused.
FP32, TF32 disabled. Environment versions are recorded; hardware/software differences may affect results.
Checkpoints every 50 epochs; less than 2GiB free causes an error. Keep adequate data-disk space.
After interruption, verify no worker remains before handling a stale lock. Never blindly remove locks.
No old process is stopped by these scripts. No Oxford job is launched outside the new partition protocol.
