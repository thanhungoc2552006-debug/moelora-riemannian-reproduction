# Functional diagnostics

| Probe | Config | Sampling | Summary |
| --- | --- | --- | --- |
| RSGD | [config](rsgd/probe_config.json) | [manifest](rsgd/sample_manifest.json) | [experts](rsgd/expert_summary.csv), [pairs](rsgd/pair_summary.csv) |
| gRSGD | [config](grsgd/probe_config.json) | [manifest](grsgd/sample_manifest.json) | [experts](grsgd/expert_summary.csv), [pairs](grsgd/pair_summary.csv) |

[Cross-method summary](grsgd/rsgd_vs_grsgd_summary.csv) ·
[Comparison figure](grsgd/figures/05_rsgd_vs_grsgd.png)

Each probe covers 64 questions, up to 16 evenly spaced question/choice tokens
per question, and v_proj/o_proj at layers 0/14/27.
The backbone is FP16 on T4; training used BF16. Original paths inside configs
are retained as historical metadata.

Raw records are in matrix_pairs.csv, functional_pairs.csv and routing_experts.csv.
Summaries are in expert_summary.csv and pair_summary.csv; plots are in figures/.

The generating script has not been supplied. These are archived measurements,
not a currently executable probe pipeline. Missing definitions are not inferred
from filenames. Output cosine or entropy alone does not establish
specialization, traffic balance or collapse.
