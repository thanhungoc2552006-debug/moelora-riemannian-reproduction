# ScienceQA: paper protocol variant, seed 42

Completed archived Drive runs: 800 updates; E=20, K=10, rank=4;
alpha=16, dropout=0.05, microbatch=2, accumulation=40 (nominal effective batch 80);
BF16 backbone/FP32 adapters; pinned model/dataset revisions; linear schedule
with horizon 2222. Both audits record zero training rows without supervision.

| Method | Test accuracy | Correct / total |
| --- | ---: | ---: |
| rsgd_seed42 | 62.54% | 1391/2224 |
| grsgd_seed42 | 69.24% | 1540/2224 |

These are single-seed text-only observations, close to the ScienceQA SGD rows
of Table 1 (62.8% RSGD, 70.1% gRSGD), not reproduction of the whole table.
The earlier batch-8 experiments remain in ../long_run/.

## Provenance and checkpoints
source_snapshot/ preserves the exact original driver, patched model, optimizer,
training helpers, revisions and package versions. Original summary/config hashes
remain unchanged. The packaged train_paper module has a new hash because
dropout is now explicit. Use a NEW output directory for new runs.
For old checkpoints, restore the original source/environment rather than
bypassing the config check.
Weights remain on [Drive](https://drive.google.com/drive/folders/1758VR30_gvLQ0eZFeFKw-W2MgDtH0Tax).
The snapshot driver saves adapters/router, optimizer, RNG, step, epoch and
next_batch. Exact interrupted/resumed equivalence was not tested in this sync.

## Diagnostics
Functional probes: 64 questions, up to 16 question/choice tokens per question,
v_proj/o_proj at layers 0/14/27; FP16 backbone on T4, while training used BF16.
See probe_config.json and sample_manifest.json for the actual sampling.
Generating scripts were absent from the supplied folder; outputs/configs are
archived, but regeneration remains incomplete.
Cosine/entropy alone does not establish expert specialization.

## Start a new run
From the repository root (requires GPU and Hugging Face access):

    python -m pip install -e ".[scienceqa]"
    python -m moelora_repro.scienceqa.train_paper --mode riemannian --out outputs/paper_v1_seed42_rsgd --revisions results/scienceqa/paper_protocol_v1/source_snapshot/revisions.json
    python -m moelora_repro.scienceqa.train_paper --mode moe-riemannian --out outputs/paper_v1_seed42_grsgd --revisions results/scienceqa/paper_protocol_v1/source_snapshot/revisions.json
