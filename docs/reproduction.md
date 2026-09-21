# Reproduction scope and protocols

Reference: [A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation Models](https://arxiv.org/abs/2502.15828),
with [the authors' source code](https://github.com/THUDM/MoELoRA_Riemannian).
This repository is an independent reproduction study in progress.

## Current protocols

| Setting | Synthetic | ScienceQA pilot | ScienceQA long run |
| --- | --- | --- | --- |
| Backbone | One frozen linear map | Llama-3.2-3B | Llama-3.2-3B |
| Data | Generated regression | First 1,000 shuffled text-only train examples; 200 validation examples | All text-only train/validation/test examples |
| Experts / rank | 8 / 4 | 20 / 4 | 20 / 4 |
| Adapter scale | 1 | alpha/rank = 8/4 | alpha/rank = 8/4 |
| Target layers | Single linear layer | q/k/v/o attention projections | q/k/v/o attention projections |
| Expert / router LR | 0.1 / 0.01 | 3e-5 / 3e-8 | 3e-5 / 3e-8 |
| Damping | 0.01 | 1e-6 | 1e-6 |
| Default seed(s) | 0/1/2; similarity: 0–4 | 42 | 42 |
| Training budget | 300 updates | At most 100 microbatches, batch size 1 | 1 epoch, microbatch 2, accumulation 4 |
| Evaluation | Full validation MSE | Loss on at most 100 validation batches; diagnostics on one fixed batch | Full validation loss; final generated-answer accuracy on text-only test set |
| Precision | FP32, no mixed precision | BF16 backbone, FP32 adapters | BF16 backbone, FP32 adapters |

ScienceQA loads `derek-thomas/ScienceQA`, filters `image is None`, formats
question/options/hint into a prompt and supervises an answer sentence. The
maximum sequence length is 256. The long run adds EOS to training targets and
generates at most 8 new tokens for final answer extraction. Pilot and long-run
formatting are intentionally kept separate because they are different historical
protocols.

The synthetic track uses paired initializations, tasks and minibatch sequences
across methods within each seed. Its assertions cover forward equality, local
expert/gate derivatives, routing, preconditioning, frozen weights and finite
updates. See [the full mechanism notes](../experiments/synthetic/mechanism.md).

## How to interpret the archived evidence

Synthetic results support the controlled mechanisms exercised by the tests and
the recorded regression setup. The benefit is not monotonic in K. A surrogate
expert backward proportional to sqrt(g) should not be described as proof that
the actual finite forward update has exactly the paper's theoretical geometry.

The two K=10 ScienceQA histories report 814 optimizer steps and final accuracy
on 2,224 text-only test examples. They are single-run observations. They do not
provide uncertainty over seeds or reproduce a multimodal/full benchmark score.
The result files are historical evidence, not output from a new run of the
reorganized package.

## Known gaps to resolve before new scientific claims

- Both existing ScienceQA training tokenizers concatenate prompt and answer
  before enforcing the sequence limit. A sufficiently long prompt can remove
  every supervised answer token, leaving labels entirely `-100`. Fix both paths,
  test long prompts explicitly, and give reruns a new experiment name. Earlier
  PR #7 addresses the pilot path only.
- Model and dataset revisions are not pinned in the historical drivers. New
  manifests record Python dependencies and the code commit, but do not make
  unpinned remote model/data revisions immutable.
- Historical artifacts lack complete provenance. For example, synthetic JSONs
  record device and dependency versions, but ScienceQA summaries omit several
  environment fields. Missing values must remain unknown rather than inferred
  as measured facts.
- The drivers save metrics, not resumable model/optimizer checkpoints. An
  interrupted long run cannot resume solely from its CSV or manifest.
- The implementation computes dense expert contributions before routing masks;
  it is not evidence of sparse-dispatch throughput or memory savings.

The layout change preserves the optimizer math, routing, tokenization and
evaluation behavior. A separate algorithm/protocol change should document its
effect on comparison with these archived results.
