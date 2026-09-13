# SOH hyperparameter search on one H100

This repository runs the complete four-dataset search for CS2, CX2, MIT/Severson, and Oxford. One GPU process is launched per dataset on the same H100.

The fixed protocol uses 500 epochs and seeds 1-10. It contains 48,640 uniquely identified training tasks. Model weights are trained only on the training cell, while hyperparameters are selected using the configuration cell. Completed tasks retain `result.json`; checkpoints are retained only for interrupted tasks and removed after successful completion.

Read [README_H100.md](README_H100.md) before running. On Windows, use `START_H100.cmd`. On Linux, prepare the documented Python/PyTorch environment and run `start_h100.sh`.

Do not edit `runner.py`, `source/`, `inputs/`, or `manifest.json` after a run begins. Their hashes define the experiment protocol.
