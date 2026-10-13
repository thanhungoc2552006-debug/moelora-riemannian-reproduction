# ScienceQA protocol and scope

Reference: [A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation
Models](https://arxiv.org/abs/2502.15828).
This study examines the ScienceQA RSGD/gRSGD pair, not all of Table 1.

## Completed protocol

| Setting | Archived seed-42 runs |
| --- | --- |
| Model/data | Llama-3.2-3B / derek-thomas/ScienceQA; pinned revisions |
| Subset | image is None; 6,508 train / 2,144 validation / 2,224 test |
| Experts / Top-K / rank | 20 / 10 / 4 |
| Alpha / dropout | 16 / 0.05 |
| Modules | q_proj, k_proj, v_proj, o_proj |
| Precision | BF16 backbone, FP32 adapters/router |
| Updates / seed | 800 / 42 |
| Microbatch / accumulation | 2 / 40; nominal batch 80 |
| Expert / gate LR | 3e-5 / 3e-8; linear decay |
| Scheduler horizon | 2,222 updates |
| Damping / gate clipping | 1e-6 / max norm 1.0 |
| Sequence / prompt budget | 256 / 240 |
| Generation | Greedy, at most 10 new tokens |
| Final evaluation | Full text-only test; no test-based checkpoint selection |

Answer/EOS space is reserved. Training and generation share the same prompt
truncation rule. Audits record zero unsupervised train rows, one truncated train
prompt and no truncated test prompt. Epoch-end accumulation groups can be smaller.

The reorganized driver preserves the newer protocol and optimizer math.
Helper/import changes alter hashes, so new executions are new runs.
[Original source](../results/scienceqa/source_snapshot/README.md).

## Interpretation limits

- One seed provides no multi-seed uncertainty estimate.
- Nearby accuracy and some matching hyperparameters do not establish an
  identical paper evaluation protocol.
- Image-dependent examples are excluded.
- Archived metric histories have gaps; plots use recorded points only.
- FP16 functional probes differ from BF16 training; probe source is not supplied.
- Checkpoint save/load exists, but exact uninterrupted/resumed equivalence
  was not demonstrated by this reorganization.
- Experts are computed before masks; this is not sparse-dispatch performance evidence.

## Synthetic scope

[The separate study](../synthetic/README.md) uses generated regression and paired
seeds to isolate Jacobians, routing and preconditioning. It distinguishes the
surrogate sqrt(g) backward from actual finite forward change. MSE is not QA accuracy.

## Historical ScienceQA

Earlier pilot/Top-K/batch-8 runs remain in
[the archive](https://github.com/thanhungoc2552006-debug/moelora-riemannian-reproduction/tree/archive/before-scienceqa-cleanup).
Their different protocols and possible target-truncation issue are not mixed
into the active comparison.
