# Results index

These are curated, committed artifacts. New runs belong in ignored `outputs/`
until their configuration, metrics and interpretation are ready to review.

## Historical experiments

| Experiment | Protocol / coverage | Key files |
| --- | --- | --- |
| [Synthetic baseline](synthetic/main/) | 3 seeds, K=1/2/4/8, two modes, 300 steps | [aggregate.csv](synthetic/main/aggregate.csv), [training loss](synthetic/main/training_loss.png) |
| [Synthetic expert similarity](synthetic/expert_similarity/) | 5 seeds, K=4/8, two modes | [tail summary](synthetic/expert_similarity/similarity_tail_summary.csv), [figure](synthetic/expert_similarity/expert_similarity.png) |
| [ScienceQA early pilot](scienceqa/pilot/) | Early validation comparison; full run provenance unavailable | [comparison](scienceqa/pilot/rsgd_vs_grsgd_scienceqa.csv) |
| [ScienceQA Top-K diagnostics](scienceqa/topk/) | K=4/8 short runs and derived comparisons | [diagnostics](scienceqa/topk/expert_diagnostics_comparison.csv) |
| [ScienceQA K=10 long run](scienceqa/long_run/) | One epoch per method, 814 optimizer steps, 2,224 text-only test examples | [final comparison](scienceqa/long_run/k10_longrun_final_comparison.csv) |

The 148 existing CSV/JSON/PNG files retain their original bytes. Directory moves
do not create new measurements. Historical source commits can be located with
`git log --follow -- <file>` after this change is committed.

The new recipes mirror driver settings, but are not historical provenance
records. Older JSON files may omit options added later. Some ScienceQA plots
and comparison tables were committed without the scripts that generated them;
their presence alone does not imply full figure reproducibility.

Read [the protocol limitations](../docs/reproduction.md) before treating the
ScienceQA scores as benchmark reproduction evidence.

## Registering a new result

Copy a reviewed run directory from `outputs/` to a descriptive directory such
as `results/scienceqa/k10_seed_sweep/seed42_rsgd/`. Include the effective recipe,
manifest, logs, metrics and figures needed to understand it. Add an experiment
README using [this template](../docs/experiment-template.md), then add one row
to this index. Keep failed runs identifiable and keep independent seeds/methods
in separate directories. Store large weights and datasets outside Git.
