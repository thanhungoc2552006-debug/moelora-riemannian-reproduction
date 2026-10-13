# ScienceQA — completed seed-42 comparison

Protocol: `scienceqa-paper-hparams-corrected-labels-bf16-v1`.
Completed 800-step text-only runs imported from Drive.

| Method | Accuracy | Correct / total | Final validation loss |
| --- | ---: | ---: | ---: |
| RSGD | 62.54% | 1,391 / 2,224 | 0.090254 |
| gRSGD | 69.24% | 1,540 / 2,224 | 0.073733 |

Improvement: 6.70 percentage points. Paper Table 1 reports 62.8% and 70.1%
for its SGD rows. See [protocol and scope](../../docs/reproduction.md).

## Evidence

| Location | Purpose |
| --- | --- |
| [RSGD run](runs/rsgd_seed42/) | Original config, summary, audit, environment, metrics and predictions |
| [gRSGD run](runs/grsgd_seed42/) | Original gRSGD records |
| [Diagnostics](diagnostics/README.md) | Functional probes and sampling manifests |
| [Source snapshot](source_snapshot/README.md) | Original source, hashes, revisions and environment |
| [comparison.csv](comparison.csv) | Rebuilt summary |
| [Validation figure](figures/validation_loss.png) | Rebuilt from recorded losses |

Original evidence files retain their bytes. Directory moves are not new runs;
absolute paths inside original metadata remain unchanged.

## Protocol and limits

6,508 train / 2,144 validation / 2,224 test examples, filtered by image is None.
E=20/K=10/r=4; alpha=16; dropout=0.05; microbatch=2, accumulation=40;
800 updates; seed=42; scheduler horizon=2222; pinned revisions.
The nominal batch is 80; accumulation groups at epoch boundaries may be smaller.
Training uses a BF16 backbone and FP32 adapters/router. Both audits record zero
training rows without supervised answer tokens.

One seed does not establish uncertainty. Validation accuracy on the first 256
examples is not full test accuracy. Archived histories omit RSGD steps 100/200
and gRSGD step 300; no points are reconstructed.
The probe backbone is FP16 and its source is not supplied.
Exact uninterrupted/resumed equivalence has not been tested in this reorganization.

## Regenerate derived files

From the repository root: `python scripts/summarize_scienceqa.py`.
It rebuilds comparison.csv and the validation-loss figure without loading a
model or modifying original run evidence.

[Checkpoint Drive folder](https://drive.google.com/drive/folders/1758VR30_gvLQ0eZFeFKw-W2MgDtH0Tax).
New runs use `moelora_repro.scienceqa.train` and new output directories.
