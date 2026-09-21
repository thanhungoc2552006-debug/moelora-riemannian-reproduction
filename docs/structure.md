# Code layout and migration

The package separates two experimental protocols. Synthetic and LLM modules
keep their own model/optimizer implementations in this organization change;
their existing mathematical behavior is preserved.

## Path migration

| Previous path | Canonical path |
| --- | --- |
| `models/moe_lora.py` | `src/moelora_repro/synthetic/model.py` |
| `optim/riemannian_sgd.py` | `src/moelora_repro/synthetic/optimizer.py` |
| `train.py` implementation | `src/moelora_repro/synthetic/train.py` |
| `experiment.py` implementation | `src/moelora_repro/synthetic/experiment.py` |
| `real_task/moe_lora.py` | `src/moelora_repro/scienceqa/model.py` |
| `real_task/riemannian_sgd.py` | `src/moelora_repro/scienceqa/optimizer.py` |
| `real_task/prepare_scienceqa.py` | `src/moelora_repro/scienceqa/data.py` |
| `real_task/diagnostics.py` | `src/moelora_repro/scienceqa/diagnostics.py` |
| `real_task/train_scienceqa.py` implementation | `src/moelora_repro/scienceqa/train_pilot.py` |
| `real_task/train_scienceqa_long.py` implementation | `src/moelora_repro/scienceqa/train.py` |
| Original synthetic README | `docs/synthetic.md` |
| `results/main/` | `results/synthetic/main/` |
| `results/expert_similarity/` | `results/synthetic/expert_similarity/` |
| Files directly in `results/real_task/` | `results/scienceqa/pilot/` |
| `results/real_task/topk/` | `results/scienceqa/topk/` |
| `results/real_task/long_run/` | `results/scienceqa/long_run/` |

All 148 pre-existing result files are moved without changing their contents.
Old result paths and old Python imports must be updated using this table.

## Commands and imports

Install with `python -m pip install -e .` from the repository root. Package
imports then work outside the source directory as well:

```python
from moelora_repro.synthetic.model import MoELoRA
from moelora_repro.synthetic.train import Config, run
from moelora_repro.scienceqa.model import inject_moe_lora
```

| Historical command | Canonical direct command |
| --- | --- |
| `python train.py ...` | `python -m moelora_repro.synthetic.train ...` |
| `python experiment.py ...` | `python -m moelora_repro.synthetic.experiment ...` |
| `python real_task/train_scienceqa.py ...` | `python -m moelora_repro.scienceqa.train_pilot ...` |
| `python real_task/train_scienceqa_long.py ...` | `python -m moelora_repro.scienceqa.train ...` |

The four historical commands still delegate to the canonical modules; the
wrappers contain no training logic and are not a compatibility layer for old
imports. Both `--top_k` and `--top-k` forms are accepted by the ScienceQA CLIs.
The new `--seed` option retains 42 as its default.

Direct commands write under `outputs/` by default and accept `--out`. They
retain the driver's original filename/overwrite behavior. Prefer
`python -m moelora_repro.run --config ...` for unique directories, overwrite
protection and automatic provenance.

## Relationship to earlier pull requests

This layout was prepared from `main` at `bdd16e7`, which already includes the
ScienceQA long-run results. It is independent of the still-open
[PR #7](https://github.com/thanhungoc2552006-debug/moelora-riemannian-reproduction/pull/7)
and [PR #8](https://github.com/thanhungoc2552006-debug/moelora-riemannian-reproduction/pull/8).
PR #8 is based on PR #7's `professionalize-reproduction` branch.

If this layout is accepted first, port the relevant changes from those PRs to
the new paths: the pilot tokenization fix belongs in `scienceqa/train_pilot.py`,
and shared optimization primitives can be placed in `moelora_repro/core.py`.
The newer long-run tokenizer also needs the supervised-target fix. Reconcile
their documentation and import changes against this structure before merging;
the old branches should not be merged blindly over the moved files.
