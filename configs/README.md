# Experiment recipes

Run from the repository root after installing the package:

```bash
python -m moelora_repro.run --config configs/synthetic/smoke.json
```

| Recipe | Purpose |
| --- | --- |
| [synthetic/smoke.json](synthetic/smoke.json) | 5-step CPU sweep for installation checks |
| [synthetic/main.json](synthetic/main.json) | 300-step, 3-seed synthetic comparison |
| [synthetic/expert_similarity.json](synthetic/expert_similarity.json) | K=4/8, 5-seed expert diagnostics |
| [scienceqa/pilot_rsgd_k4.json](scienceqa/pilot_rsgd_k4.json) | 100-step RSGD pilot |
| [scienceqa/pilot_grsgd_k4.json](scienceqa/pilot_grsgd_k4.json) | Paired gRSGD pilot |
| [scienceqa/long_rsgd_k10.json](scienceqa/long_rsgd_k10.json) | RSGD, K=10, one epoch, effective batch 8 |
| [scienceqa/long_grsgd_k10.json](scienceqa/long_grsgd_k10.json) | Paired gRSGD long run |

A recipe contains exactly `name`, `entrypoint`, and `parameters`. Names use
lowercase letters, digits, hyphens or underscores. Parameter keys use
underscores and map to CLI flags, for example `top_k` becomes `--top-k`.
Values may be strings, numbers, lists of these, or `null` (omit the flag).
Driver argument parsers reject unknown flags when a run starts.

`--set KEY=VALUE` accepts JSON values, falling back to a string. For example,
`--set steps=50`, `--set device=cuda`, or `--set 'seeds=[0,1,2,3,4]'`.
`--dry-run` checks the recipe format and displays overrides; it does not check
GPU capacity, model access, or download datasets.

Use `--out` on the runner to choose a fresh output directory. Do not put `out`
inside a recipe. Relative config/output paths are relative to the current
working directory, so documented commands assume the repository root.

ScienceQA's model, rank, expert count, learning rates and tokenization protocol
remain the existing driver constants. Their values are documented in
[the protocol notes](../docs/reproduction.md) and tied to the saved code commit.
The pilot iterates over its 1,000-example training subset once, so its
`num_steps` is a cap, not an instruction to repeat the data.

Recipes were reconstructed from the checked-in drivers and result metadata.
They are starting points for future runs, not evidence that historical results
were produced through this runner. Historical artifacts have no retroactively
invented manifests.
