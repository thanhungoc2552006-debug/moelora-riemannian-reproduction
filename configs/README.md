# LLM experiment recipes

These recipes configure the main text-only ScienceQA reproduction. Run from the
repository root after installing `python -m pip install -e ".[scienceqa]"`.

| Recipe | Purpose |
| --- | --- |
| [scienceqa/pilot_rsgd_k4.json](scienceqa/pilot_rsgd_k4.json) | 100-step RSGD pilot |
| [scienceqa/pilot_grsgd_k4.json](scienceqa/pilot_grsgd_k4.json) | Paired gRSGD pilot |
| [scienceqa/long_rsgd_k10.json](scienceqa/long_rsgd_k10.json) | RSGD, K=10, one epoch, effective batch 8 |
| [scienceqa/long_grsgd_k10.json](scienceqa/long_grsgd_k10.json) | Paired gRSGD long run |

```bash
python -m moelora_repro.run --config configs/scienceqa/long_grsgd_k10.json --dry-run
```

A recipe contains exactly `name`, `entrypoint`, and `parameters`. Names use
lowercase letters, digits, hyphens or underscores. Parameter keys use
underscores and map to CLI flags: `top_k` becomes `--top-k`. Values may be strings,
numbers, lists of these, or `null` (omit the flag).

`--set KEY=VALUE` accepts JSON values, falling back to a string, for example
`--set top_k=8` or `--set seed=43`. The driver rejects unknown flags when a run
starts. `--dry-run` validates recipe format and displays the effective values;
it does not check GPU capacity or model access.

Use the launcher's `--out` to choose a new output directory; do not put `out`
inside a recipe. Relative paths are interpreted from the current working directory.

The model, rank, expert count, learning rates and tokenization protocol remain
the existing driver constants, documented in [the protocol notes](../docs/reproduction.md)
and tied to the saved code commit. The pilot iterates over its 1,000-example
training subset once; `num_steps` is a cap rather than a request to repeat data.

These recipes reflect the checked-in drivers and metadata, but are not
retroactively created provenance for historical results.

Synthetic recipes live with their supporting study in
[experiments/synthetic/configs/](../experiments/synthetic/configs/), with usage
instructions in [its README](../experiments/synthetic/README.md).
