# MoE-LoRA Riemannian Reproduction

An independent reproduction study of **RSGD and gate-rescaled gRSGD for LLM
fine-tuning**, based on [A Stronger Mixture of Low-Rank Experts for Fine-Tuning
Foundation Models](https://arxiv.org/abs/2502.15828).
The authors' implementation is [THUDM/MoELoRA_Riemannian](https://github.com/THUDM/MoELoRA_Riemannian).

The main experiment adapts **Llama-3.2-3B on text-only ScienceQA**. Controlled
synthetic experiments provide supporting checks of gradients, routing and
optimization; their code, recipes, results and explanations are collected in
[experiments/synthetic/](experiments/synthetic/README.md).

**Status: reproduction in progress.** These local protocols and historical
results do not establish reproduction of the paper's benchmark scores. The
existing ScienceQA target-truncation issue and other protocol gaps are described
in [the reproduction notes](docs/reproduction.md).

[Hướng dẫn tiếng Việt](docs/workflow.vi.md) ·
[LLM results](results/README.md) ·
[Protocols and limitations](docs/reproduction.md) ·
[Code layout](docs/structure.md)

## Run the LLM experiments

Python **3.10+**. From the repository root, preferably inside a virtual environment:

```bash
python -m pip install -e ".[scienceqa]"
```

Install the appropriate CUDA build of PyTorch for your machine. ScienceQA
training requires access to `meta-llama/Llama-3.2-3B` on Hugging Face and sufficient
GPU memory; the long-run driver explicitly requires CUDA. Keep access tokens in
your local Hugging Face login, outside this repository.

First inspect the effective configuration without downloading data or training:

```bash
python -m moelora_repro.run --config configs/scienceqa/long_grsgd_k10.json --dry-run
```

Run either method on a prepared GPU machine:

```bash
python -m moelora_repro.run --config configs/scienceqa/long_rsgd_k10.json
python -m moelora_repro.run --config configs/scienceqa/long_grsgd_k10.json
```

The pilot recipes run a shorter experiment. Override a supported parameter
without copying training code:

```bash
python -m moelora_repro.run --config configs/scienceqa/pilot_grsgd_k4.json --set top_k=8 --set seed=43
```

Each run creates a fresh `outputs/<recipe>/<run-id>/` directory with the effective
`recipe.json`, `manifest.json`, `console.log` and driver outputs. The manifest
records code/recipe commits, dirty-worktree flags, installed dependency versions,
command, timestamps and exit status. An explicit `--out` must name a new directory.
The installed `moelora-run` command is equivalent to `python -m moelora_repro.run`.

## Recorded LLM results

The archived one-epoch ScienceQA K=10 runs use 20 experts, rank 4, microbatch 2,
gradient accumulation 4 and 814 optimizer steps. The text-only test set contains
2,224 examples.

| Method | Final validation loss | Test accuracy | Correct / total |
| --- | ---: | ---: | ---: |
| RSGD | 0.10825 | 71.18% | 1,583 / 2,224 |
| gRSGD | 0.08315 | 77.79% | 1,730 / 2,224 |

Source: [committed final comparison](results/scienceqa/long_run/k10_longrun_final_comparison.csv).
These are one-run historical observations with no multi-seed uncertainty estimate.
They were preserved during the repository reorganization, not regenerated.
See [the results index](results/README.md) for the pilot and Top-K diagnostics.

## Repository layout

| Location | Purpose |
| --- | --- |
| `src/moelora_repro/scienceqa/` | Main LLM implementation: adapters, optimization, data, diagnostics and training |
| `src/moelora_repro/run.py` | Common experiment launcher and provenance recording |
| `configs/scienceqa/` | LLM pilot and long-run recipes |
| `results/scienceqa/` | Curated historical LLM results |
| `experiments/synthetic/` | Supporting synthetic study, including its code, configs, results and documentation |
| `docs/` | Protocols, migration guide and contributor workflow |
| `tests/` | Mechanism and workflow checks |
| `outputs/` | Local run outputs, ignored by Git |

`pyproject.toml` is the single source for package installation and dependencies.
The old root training scripts and `real_task/` wrappers have been removed.
Use the recipe launcher or the [documented module commands](docs/structure.md).

## Supporting synthetic checks

The small regression experiments isolate routing and optimizer behavior without
loading an LLM. Everything needed to inspect that study is in
[experiments/synthetic/](experiments/synthetic/README.md), including its two
archived result sets and detailed mechanism notes.

A CPU-only check from the repository root:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m moelora_repro.run --config experiments/synthetic/configs/smoke.json
```

## Add the next LLM experiment

1. Create a named recipe in `configs/scienceqa/` and commit the code/configuration.
2. Run it through the launcher and inspect the saved status, metrics and diagnostics.
3. Copy a reviewed run into `results/scienceqa/<experiment>/` with its provenance.
4. Add a short interpretation using the [experiment template](docs/experiment-template.md)
   and register it in the [LLM results index](results/README.md).

Supporting synthetic recipes and curated results stay inside
`experiments/synthetic/`. See [the Vietnamese workflow](docs/workflow.vi.md).
