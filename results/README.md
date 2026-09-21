# LLM results

This directory contains the main text-only ScienceQA experiments. New runs
belong in ignored `outputs/` until their configuration, metrics and interpretation
are ready to review.

| Experiment | Protocol / coverage | Key files |
| --- | --- | --- |
| [Early pilot](scienceqa/pilot/) | Early validation comparison; full provenance unavailable | [comparison](scienceqa/pilot/rsgd_vs_grsgd_scienceqa.csv) |
| [Top-K diagnostics](scienceqa/topk/) | K=4/8 short runs and derived comparisons | [diagnostics](scienceqa/topk/expert_diagnostics_comparison.csv) |
| [K=10 long run](scienceqa/long_run/) | One epoch per method, 814 optimizer steps, 2,224 text-only test examples | [final comparison](scienceqa/long_run/k10_longrun_final_comparison.csv) |

The 25 existing ScienceQA CSV/JSON/PNG files retain their original bytes.
Directory organization is not a new measurement. Historical source commits
can be located with `git log --follow -- <file>`.

Some plots and comparison tables were committed without their generating
scripts, and historical summaries lack complete run provenance. The new recipes
mirror driver settings but are not records of the original execution environment.
Read [the protocol limitations](../docs/reproduction.md) before interpreting
these scores as benchmark reproduction evidence.

## Registering a new LLM result

Copy a reviewed run into a directory such as
`results/scienceqa/k10_seed_sweep/seed42_rsgd/`. Keep the effective recipe,
manifest, logs, metrics and figures needed to interpret it. Add a README using
[the experiment template](../docs/experiment-template.md), then register it here.
Keep independent seeds/methods separate and large weights/datasets outside Git.

The supporting synthetic results are collected separately in
[experiments/synthetic/results/](../experiments/synthetic/results/README.md).
