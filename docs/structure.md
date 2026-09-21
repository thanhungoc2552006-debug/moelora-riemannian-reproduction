# Code layout

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

## Installation and commands

From the repository root:

```bash
python -m pip install -e ".[scienceqa]"
```

For only the synthetic checks, `python -m pip install -e .` is sufficient.

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

## Run outputs and curated results

New runs are written to ignored `outputs/<recipe>/<run-id>/` directories.
Keep reviewed LLM evidence in `results/scienceqa/` and supporting synthetic
artifacts in `experiments/synthetic/results/`. See the
[contributor workflow](workflow.vi.md) for recording a new experiment.

For retired paths, removed wrappers and older development branches, consult
[repository history and migration](history.md).
