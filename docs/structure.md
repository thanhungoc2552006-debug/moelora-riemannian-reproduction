# Repository structure

## ScienceQA

| Location | Responsibility |
| --- | --- |
| `src/moelora_repro/scienceqa/model.py` | Adapters, routing and configurable dropout |
| `src/moelora_repro/scienceqa/optimizer.py` | Riemannian expert optimizer |
| `src/moelora_repro/scienceqa/data.py` | Unchanged seed, padding and answer-parsing helpers |
| `src/moelora_repro/scienceqa/train.py` | Corrected-label protocol, evaluation and resumable training |
| `src/moelora_repro/run.py` | Recipe launcher |
| `configs/scienceqa/` | Current recipes/revisions |
| `results/scienceqa/` | Runs, probes, snapshots and derived comparison |
| `scripts/summarize_scienceqa.py` | Rebuild summary CSV and validation figure |

The newer train_paper.py is now the canonical train.py. Earlier drivers,
recipes and ScienceQA outputs have left the active tree. Original source
snapshots are unchanged, including historical paths.
[Historical archive](https://github.com/thanhungoc2552006-debug/moelora-riemannian-reproduction/tree/archive/before-scienceqa-cleanup).

## Synthetic

[synthetic/](../synthetic/README.md) contains all synthetic-specific code,
recipes, results, notes and mechanism tests. Setuptools maps it to import name
`moelora_synthetic`; its math and result bytes are unchanged.

## Commands

| Purpose | Entry point |
| --- | --- |
| New run with provenance | `python -m moelora_repro.run --config <recipe.json>` |
| ScienceQA direct/resume | `python -m moelora_repro.scienceqa.train ...` |
| Synthetic training | `python -m moelora_synthetic.train ...` |
| Synthetic sweep | `python -m moelora_synthetic.experiment ...` |
| ScienceQA/workflow checks | `python -m unittest discover -s tests -v` |
| Synthetic checks | `python -m unittest discover -s synthetic/tests -v` |

Run from the repository root. Reinstall `pip install -e .` or
`pip install -e ".[scienceqa]"` after this relocation.
The launcher creates new output directories. Direct training resumes only if
saved config/source/environment match.
