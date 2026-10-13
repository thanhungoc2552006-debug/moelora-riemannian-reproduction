# Original source and checkpoint provenance

These files preserve the original training source; they are evidence, not
active package entrypoints.

| File | Role |
| --- | --- |
| [scienceqa_colab_driver.py](scienceqa_colab_driver.py) | Original driver and checkpoint save/load |
| [model.py](model.py) | Exact dropout-patched model |
| [optimizer.py](optimizer.py) | Exact expert optimizer |
| [train_helpers.py](train_helpers.py) | Original helper module |
| [colab_source.diff](colab_source.diff) | Original model patch |
| [revisions.json](revisions.json) | Model/data revisions |
| [pip_freeze.txt](pip_freeze.txt) | Original package versions |

Source commit and SHA256 hashes are recorded in
[RSGD config](../runs/rsgd_seed42/config.json) and
[gRSGD config](../runs/grsgd_seed42/config.json).

The driver saves trainable parameters, gate optimizer state, expert LR/reg,
Python/NumPy/Torch/CUDA RNG, step, epoch and next batch.
[Weights on Drive](https://drive.google.com/drive/folders/1758VR30_gvLQ0eZFeFKw-W2MgDtH0Tax).

To resume original weights, restore the exact source/package/environment.
Executing this driver against a newer installed package does not restore that
source. The reorganized driver has new hashes: new executions use fresh output
directories; configuration guards remain enabled.
