# MoE-LoRA Riemannian Reproduction

An independent, ongoing reproduction study of **RSGD and gate-rescaled gRSGD**
for mixtures of low-rank experts, based on
[A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation Models](https://arxiv.org/abs/2502.15828).
The authors' implementation is [THUDM/MoELoRA_Riemannian](https://github.com/THUDM/MoELoRA_Riemannian).

This repository contains controlled synthetic experiments and a text-only
ScienceQA extension using Llama-3.2-3B. The stored results describe these local
protocols; they do not establish reproduction of the paper's benchmark scores.

[Hướng dẫn làm việc bằng tiếng Việt](docs/workflow.vi.md) ·
[Experiment protocols](docs/reproduction.md) ·
[Results index](results/README.md) ·
[Code layout and migration](docs/structure.md)

## Experiment tracks

| Track | Purpose | Archived evidence |
| --- | --- | --- |
| Synthetic regression | Check routing, expert gradients, and paired RSGD/gRSGD updates | 3 seeds × 4 values of K × 2 modes |
| Synthetic expert similarity | Compare expert output, BA, and gradient cosine at K=4/8 | 5 seeds × 2 values of K × 2 modes |
| ScienceQA pilot | Short text-only training and expert diagnostics | Pilot and K=4/8 CSVs and figures |
| ScienceQA long run | One-epoch K=10 training and generated-answer accuracy | Two histories and final summaries, 814 optimizer steps each |

## Install

Python **3.10+**. Run these commands from the repository root, preferably in a
virtual environment. Install the appropriate CUDA build of PyTorch first if
you will train on a GPU.

```bash
git clone https://github.com/thanhungoc2552006-debug/moelora-riemannian-reproduction.git
cd moelora-riemannian-reproduction
python -m pip install -e .
```

The base install runs synthetic experiments without downloading a model or
dataset. For ScienceQA, also install the optional dependencies:

```bash
python -m pip install -e ".[scienceqa]"
```

ScienceQA training needs access to `meta-llama/Llama-3.2-3B` on Hugging Face and
sufficient GPU memory. The long-run driver requires CUDA. Keep model access
tokens in your local Hugging Face login, outside the repository.

`requirements.txt` and `requirements-scienceqa.txt` remain available as install
shortcuts. All dependency definitions are maintained in `pyproject.toml`.

## Quick start

Run the CPU tests and a small end-to-end sweep:

```bash
python -m unittest discover -s tests -v
python -m moelora_repro.run --config configs/synthetic/smoke.json
```

Run a complete synthetic recipe or inspect a ScienceQA recipe:

```bash
python -m moelora_repro.run --config configs/synthetic/main.json
python -m moelora_repro.run --config configs/synthetic/expert_similarity.json
python -m moelora_repro.run --config configs/scienceqa/long_grsgd_k10.json --dry-run
```

Override a setting without editing or duplicating training code:

```bash
python -m moelora_repro.run --config configs/synthetic/main.json --set device=cpu --set steps=100
python -m moelora_repro.run --config configs/scienceqa/pilot_rsgd_k4.json --set top_k=8 --set seed=43
```

Each recipe invocation creates a fresh `outputs/<recipe>/<run-id>/` directory
containing `recipe.json`, `manifest.json`, `console.log`, and the driver's
CSV/JSON/figure outputs. The manifest records the code and recipe Git commits,
dirty-worktree flags, installed dependency versions, command, timestamps, and
exit status. `--out` can choose another **new** directory; existing directories
are refused. `--dry-run` only displays the effective recipe.

The installed `moelora-run` command is equivalent to `python -m moelora_repro.run`.

## Repository layout

| Location | What belongs here |
| --- | --- |
| `src/moelora_repro/synthetic/` | Synthetic model, optimizer, training, sweeps and plots |
| `src/moelora_repro/scienceqa/` | ScienceQA model injection, data, diagnostics, pilot and long-run training |
| `configs/` | Named, runnable JSON experiment recipes |
| `results/synthetic/` | Curated synthetic results already committed to Git |
| `results/scienceqa/` | Curated pilot, Top-K and long-run ScienceQA results |
| `docs/` | Protocols, scientific interpretation and contributor workflow |
| `tests/` | Mechanism regression tests and experiment-runner checks |
| `outputs/` | Local runs and logs; ignored by Git |

The four historical training commands remain as small compatibility wrappers
after installing the package. New commands and imports should use the package;
see the [migration table](docs/structure.md).

## Recorded results

Synthetic validation MSE, taken from the committed
[aggregate CSV](results/synthetic/main/aggregate.csv):

| K | RSGD | gRSGD | Mean paired MSE reduction |
| --- | ---: | ---: | ---: |
| 1 | 0.73915 | 0.73915 | 0.00% |
| 2 | 0.57654 | 0.54514 | 5.35% |
| 4 | 0.64738 | 0.58914 | 8.98% |
| 8 | 0.78362 | 0.74844 | 4.44% |

The archived ScienceQA K=10 long-run summaries report **71.18%** for RSGD
(1,583/2,224) and **77.79%** for gRSGD (1,730/2,224). See the
[final comparison](results/scienceqa/long_run/k10_longrun_final_comparison.csv).
These are one-run historical observations, with no multi-seed uncertainty estimate.

Historical artifacts are preserved as recorded. In particular, the ScienceQA
training tokenizers can truncate away supervised answer tokens for long prompts;
this existing limitation must be resolved and rerun before stronger benchmark
claims. The [protocol notes](docs/reproduction.md) separate verified mechanism
checks, historical measurements, and remaining reproduction gaps.

## Adding the next experiment

1. Copy the nearest recipe into `configs/` and give it a descriptive name.
2. Commit the code and recipe; run it with the recipe runner.
3. Inspect the manifest, metrics and diagnostics.
4. Copy a complete result folder into `results/<track>/<experiment>/`, add a
   short interpretation using the [result template](docs/experiment-template.md),
   and update the [results index](results/README.md).

Detailed equations and the original synthetic research notes are retained in
[docs/synthetic.md](docs/synthetic.md).
