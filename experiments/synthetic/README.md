# Supporting synthetic mechanism experiments

This directory collects the entire synthetic study: its model, optimizer,
training/sweep scripts, configurations, archived results and mechanism notes.
The main repository focuses on [LLM fine-tuning on ScienceQA](../../README.md).

The synthetic setup uses a small frozen linear map, LoRA experts and generated
regression data. It checks local gradients, routing invariants and optimizer
updates without downloading a model or dataset. It supports the LLM study but
does not reproduce the paper's benchmark scores.

## Contents

| Path | Purpose |
| --- | --- |
| [model.py](model.py) | MoE-LoRA forward, Top-K router and expert backward rules |
| [optimizer.py](optimizer.py) | Damped Riemannian preconditioning and SGD updates |
| [train.py](train.py) | One paired, reproducible synthetic training run |
| [experiment.py](experiment.py) | K/method/seed sweeps, aggregation and plots |
| [configs/](configs/) | Named synthetic experiment recipes |
| [results/](results/README.md) | Archived baseline and expert-similarity evidence |
| [mechanism.md](mechanism.md) | Equations, protocol details and interpretation |

## Run from the repository root

```bash
python -m pip install -e .
python -m moelora_repro.run --config experiments/synthetic/configs/smoke.json
python -m moelora_repro.run --config experiments/synthetic/configs/main.json
python -m moelora_repro.run --config experiments/synthetic/configs/expert_similarity.json
```

| Recipe | Settings |
| --- | --- |
| [smoke.json](configs/smoke.json) | 5-step CPU check, one seed, K=1/2/4 |
| [main.json](configs/main.json) | 300 steps, seeds 0/1/2, K=1/2/4/8 |
| [expert_similarity.json](configs/expert_similarity.json) | 300 steps, seeds 0–4, K=4/8 |

For a smaller run, use `--set steps=50 --set device=cpu`. To inspect without
training, add `--dry-run`. Every launcher invocation owns a fresh directory
under the repository's ignored `outputs/` and records configuration, versions,
commit and status. Curate reviewed outputs into this directory's `results/`.

The project installation maps these Python files to the package name
`moelora_synthetic`; no duplicate implementation is kept under `src/`.
For direct module calls, use:

```bash
python -m moelora_synthetic.train --mode moe-riemannian --top-k 4 --steps 300 --device cpu --out outputs/synthetic/single
python -m moelora_synthetic.experiment --help
```

Direct driver calls retain fixed output names and do not add a launcher manifest;
the recipe commands above are the recommended workflow. Package wheels include
the Python modules, while configs and historical results remain in the repository.

## Archived evidence

| Experiment | Coverage | Evidence |
| --- | --- | --- |
| Baseline | 3 seeds × 4 K values × 2 modes | [aggregate](results/main/aggregate.csv), [loss figure](results/main/training_loss.png) |
| Expert similarity | 5 seeds × K=4/8 × 2 modes | [tail summary](results/expert_similarity/similarity_tail_summary.csv), [figure](results/expert_similarity/expert_similarity.png) |

The 123 existing synthetic CSV/JSON/PNG files keep their original contents.
The new recipe paths do not imply that the historical data was generated with
this launcher. Read [the mechanism notes](mechanism.md) for interpretation.

Mechanism tests remain in the common [tests directory](../../tests/test_mechanism.py)
so repository CI can exercise the complete workflow with one test command.
