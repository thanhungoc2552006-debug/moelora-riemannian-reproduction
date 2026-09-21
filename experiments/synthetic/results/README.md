# Synthetic results

These are the archived results for the [supporting synthetic study](../README.md).
New unreviewed runs are written under the repository's ignored `outputs/`.

| Directory | Coverage | Key files |
| --- | --- | --- |
| [main/](main/) | 3 seeds, K=1/2/4/8, two methods, 300 steps | [aggregate.csv](main/aggregate.csv), [training_loss.png](main/training_loss.png) |
| [expert_similarity/](expert_similarity/) | 5 seeds, K=4/8, two methods | [tail summary](expert_similarity/similarity_tail_summary.csv), [figure](expert_similarity/expert_similarity.png) |

All 123 historical CSV/JSON/PNG files retain their original bytes. Older metadata
may omit options that were introduced later. The checked-in recipes are not
retroactive provenance for these old runs.

For a new study, copy the reviewed run and its recipe/manifest/logs into a named
subdirectory here, then add an interpretation using the
[experiment template](../../../docs/experiment-template.md). Register it in this
index; LLM results have their [own index](../../../results/README.md).
