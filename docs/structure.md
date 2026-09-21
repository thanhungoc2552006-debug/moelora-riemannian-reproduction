# Layout and migration

The repository foregrounds LLM fine-tuning. Main code lives in
`src/moelora_repro/`, LLM recipes in `configs/`, and curated LLM evidence in
`results/`. The supporting synthetic study is collected in
[experiments/synthetic/](../experiments/synthetic/README.md).

## Main and supporting code

| Location | Responsibility |
| --- | --- |
| `src/moelora_repro/scienceqa/` | LLM adapters, optimizer, diagnostics, pilot and long-run training |
| `src/moelora_repro/run.py` | Shared recipe launcher and provenance recording |
| `experiments/synthetic/` | Synthetic-specific code, configs, results and mechanism notes |
| `tests/` | Common mechanism and workflow checks |

Synthetic modules are installed as `moelora_synthetic`; main modules remain
`moelora_repro`. The mapping is declared in `pyproject.toml`. Code is not copied
between these locations. Installation includes the Python modules, while recipes
and archived artifacts remain repository resources.

The earlier mathematical implementations are preserved. Consolidating the
scientific primitives from pending PR #8 remains separate from this layout change.
If those primitives are extracted, place their shared implementation in
`src/moelora_repro/core.py` and import it from each track.

## Migration from the first layout in this PR

| Previous location | Current location |
| --- | --- |
| `src/moelora_repro/synthetic/` | `experiments/synthetic/` |
| `configs/synthetic/` | `experiments/synthetic/configs/` |
| `results/synthetic/` | `experiments/synthetic/results/` |
| `docs/synthetic.md` | `experiments/synthetic/mechanism.md` |
| Root `train.py` and `experiment.py` wrappers | Removed; use recipe or module commands below |
| `real_task/` wrappers | Removed; use the ScienceQA modules below |
| `requirements.txt`, `requirements-scienceqa.txt` | Removed; install directly from `pyproject.toml` |

All 148 historical result files are preserved: 123 synthetic artifacts are now
inside the supporting study, and 25 ScienceQA artifacts remain in `results/`.

## Migration from the original main branch

| Original path | Current implementation |
| --- | --- |
| `models/moe_lora.py` | `experiments/synthetic/model.py` |
| `optim/riemannian_sgd.py` | `experiments/synthetic/optimizer.py` |
| `train.py` | `experiments/synthetic/train.py` |
| `experiment.py` | `experiments/synthetic/experiment.py` |
| `real_task/moe_lora.py` | `src/moelora_repro/scienceqa/model.py` |
| `real_task/riemannian_sgd.py` | `src/moelora_repro/scienceqa/optimizer.py` |
| `real_task/prepare_scienceqa.py` | `src/moelora_repro/scienceqa/data.py` |
| `real_task/diagnostics.py` | `src/moelora_repro/scienceqa/diagnostics.py` |
| `real_task/train_scienceqa.py` | `src/moelora_repro/scienceqa/train_pilot.py` |
| `real_task/train_scienceqa_long.py` | `src/moelora_repro/scienceqa/train.py` |
| `results/main/` | `experiments/synthetic/results/main/` |
| `results/expert_similarity/` | `experiments/synthetic/results/expert_similarity/` |
| Files directly in `results/real_task/` | `results/scienceqa/pilot/` |
| `results/real_task/topk/` and `long_run/` | `results/scienceqa/topk/` and `long_run/` |

## Installation and commands

From the repository root:

```bash
python -m pip install -e ".[scienceqa]"
```

For only the synthetic checks, `python -m pip install -e .` is sufficient.
If you installed the previous layout in editable mode, rerun the appropriate
installation command above to register the relocated synthetic package.

| Purpose | Direct command |
| --- | --- |
| One synthetic run | `python -m moelora_synthetic.train ...` |
| Synthetic sweep | `python -m moelora_synthetic.experiment ...` |
| ScienceQA pilot | `python -m moelora_repro.scienceqa.train_pilot ...` |
| ScienceQA long run | `python -m moelora_repro.scienceqa.train ...` |

Direct commands keep the driver's original output naming and overwrite behavior.
Prefer `python -m moelora_repro.run --config <recipe.json>` for new directories,
overwrite protection and automatic provenance. ScienceQA accepts both underscore
and hyphen flags, such as `--top_k` and `--top-k`.

```python
from moelora_repro.scienceqa.model import inject_moe_lora
from moelora_synthetic.model import MoELoRA
from moelora_synthetic.train import Config, run
```

## Relationship to previous PRs

This draft was based on main at `bdd16e7`, including the later ScienceQA long-run
results. It is independent of [PR #7](https://github.com/thanhungoc2552006-debug/moelora-riemannian-reproduction/pull/7)
and [PR #8](https://github.com/thanhungoc2552006-debug/moelora-riemannian-reproduction/pull/8),
which were neither merged nor closed by this work. PR #8 is based on PR #7.

Port their relevant changes to the paths above before combining them. The pilot
target-truncation fix belongs in `scienceqa/train_pilot.py`; the newer long-run
tokenizer also needs the corresponding fix. Reconcile documentation and imports
against this layout rather than merging old paths blindly.
