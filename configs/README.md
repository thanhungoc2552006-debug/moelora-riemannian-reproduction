# ScienceQA recipes

These are new-run recipes, not retroactive provenance for archived experiments.

| File | Purpose |
| --- | --- |
| [rsgd_k10_seed42.json](scienceqa/rsgd_k10_seed42.json) | RSGD, E20/K10/r4, 800 updates, nominal batch 80 |
| [grsgd_k10_seed42.json](scienceqa/grsgd_k10_seed42.json) | Paired gRSGD setting |
| [smoke.json](scienceqa/smoke.json) | Two-update GPU/data check; no test score |
| [revisions.json](scienceqa/revisions.json) | Pinned model/data revisions |

```bash
python -m moelora_repro.run --config configs/scienceqa/grsgd_k10_seed42.json --dry-run
python -m moelora_repro.run --config configs/scienceqa/grsgd_k10_seed42.json --set seed=43
```

Run from the repository root. Recipes contain exactly name, entrypoint and
parameters. Keys become hyphenated flags; scalar/list values become arguments.
Boolean true includes a flag; false omits it. `--out` chooses a NEW directory.
Relative paths are resolved from the working directory.
Dry-run validates recipe format, not GPU/model access.

Model structure, alpha, dropout, LR/scheduler and formatting are driver constants
recorded in saved configs. CLI parameters include seed, steps, microbatch,
accumulation and data limits. Read [the protocol](../docs/reproduction.md).

Synthetic recipes live in [synthetic/configs/](../synthetic/configs/).
