# Repository history and migration

This reference is for contributors porting older code or commands. For the
current organization and entry points, see [Code layout](structure.md).

## Research layout reorganization

[PR #13](https://github.com/thanhungoc2552006-debug/moelora-riemannian-reproduction/pull/13)
separated the main LLM reproduction from its supporting synthetic study.
The reorganization preserved training behavior and all historical result files.
Mathematical primitive consolidation was left separate; shared primitives, if
extracted later, can live in `src/moelora_repro/core.py` and serve both tracks.

## Migration from the first layout iteration

| Previous location | Current location |
| --- | --- |
| `src/moelora_repro/synthetic/` | `experiments/synthetic/` |
| `configs/synthetic/` | `experiments/synthetic/configs/` |
| `results/synthetic/` | `experiments/synthetic/results/` |
| `docs/synthetic.md` | `experiments/synthetic/mechanism.md` |
| Root `train.py` and `experiment.py` wrappers | Removed; use [recipe or module commands](structure.md#installation-and-commands) |
| `real_task/` wrappers | Removed; use the [ScienceQA modules](structure.md#installation-and-commands) |
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

## Related development branches

PR #13 was based on main at `bdd16e7`, including the later ScienceQA long-run
results, and was merged on 2026-09-21. At the time of the reorganization, it was independent of [PR #7](https://github.com/thanhungoc2552006-debug/moelora-riemannian-reproduction/pull/7)
and [PR #8](https://github.com/thanhungoc2552006-debug/moelora-riemannian-reproduction/pull/8),
which were neither merged nor closed by that work. PR #8 was based on PR #7.
These notes describe the refactor context, not the current status of those PRs.

Port their relevant changes to the paths above before combining them. The pilot
target-truncation fix belongs in `scienceqa/train_pilot.py`; the newer long-run
tokenizer also needs the corresponding fix. Reconcile documentation and imports
against this layout rather than merging old paths blindly.

## Updating an older editable installation

Rerun `python -m pip install -e ".[scienceqa]"`, or `python -m pip install -e .`
for synthetic checks only, to register the relocated `moelora_synthetic` package.
