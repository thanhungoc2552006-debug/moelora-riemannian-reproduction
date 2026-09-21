# MoE-LoRA Riemannian Optimization — Reproduction Study

Independent PyTorch reproduction study of **RSGD** and **gate-rescaled RSGD (gRSGD)** for Mixture-of-Experts LoRA, based on *A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation Models* (Sun et al., ICML 2025).

> **Scope.** This repository reproduces the optimization mechanism in controlled synthetic experiments and contains a small-scale ScienceQA/Llama-3.2-3B experiment. It is **not yet a full benchmark reproduction** of all results reported in the paper.

## Reproduction status

| Component | Status | Notes |
| --- | --- | --- |
| MoE-LoRA Top-K routing | ✅ | Independent PyTorch implementation |
| Riemannian preconditioning | ✅ | Linear solves instead of explicit matrix inverses |
| Gate-gradient rescaling | ✅ | Forward-preserving detach trick |
| Synthetic controlled experiment | ✅ | Paired seeds and Top-K ablation |
| Expert-similarity diagnostics | ✅ | Output, effective LoRA map, gradient/update diagnostics |
| ScienceQA + Llama-3.2-3B pilot | 🧪 | Text-only subset; loss-based validation |
| Full paper benchmark reproduction | ⏳ | Not claimed by this repository |

## Why this repository exists

The paper identifies an optimization issue in MoE-LoRA: an expert's contribution is multiplied by its routing gate in the forward pass, and the same gate also scales the gradient flowing into that expert. The proposed gate-rescaled backward changes the local expert gradient scaling from \(g\) to \(\sqrt{g}\) while preserving the forward value.

This repository asks three practical questions:

1. Can the forward-preserving gradient trick be reproduced independently?
2. Does it change optimization behavior under controlled Top-K routing?
3. Do the same mechanisms remain observable in a small real-task LLM experiment?

## Method

For an expert output \(E_i(x)\) and selected gate \(g_i\), standard routing uses

```text
g_i * E_i(x)
```

The gRSGD implementation uses

```python
sqrt_g_const = torch.sqrt(g).detach()
expert_const = expert_output.detach()
weighted_output = (
    sqrt_g_const * expert_output
    + (g - sqrt_g_const) * expert_const
)
```

The forward value is unchanged. At the same model state and upstream gradient, the derivative with respect to the expert output changes from \(g\) to \(\sqrt{g}\), while the derivative with respect to the gate remains the same.

The Riemannian preconditioner for LoRA factors \(A_i, B_i\) is implemented as

```text
dA_i = solve(B_i^T B_i + lambda I, grad_A_i)
dB_i = solve(A_i A_i^T + lambda I, grad_B_i^T)^T

A_i <- A_i - lr * dA_i
B_i <- B_i - lr * dB_i
```

Both directions are computed from the same pre-update factors.

## Repository layout

```text
.
├── models/
│   └── moe_lora.py              # Synthetic MoE-LoRA implementation
├── optim/
│   └── riemannian_sgd.py        # Synthetic Riemannian optimizer
├── real_task/
│   ├── diagnostics.py           # Expert/router diagnostics
│   ├── moe_lora.py              # LLM MoE-LoRA injection
│   ├── prepare_scienceqa.py     # Text-only ScienceQA preparation
│   ├── riemannian_sgd.py        # LLM Riemannian optimizer
│   └── train_scienceqa.py       # Llama-3.2-3B pilot
├── tests/
│   └── test_mechanism.py
├── results/
│   ├── main/
│   ├── expert_similarity/
│   └── real_task/
├── experiment.py
├── train.py
└── requirements.txt
```

The synthetic and LLM implementations are intentionally still separated. A future refactor can unify their common core after the experimental protocol is stabilized.

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/thanhungoc2552006-debug/moelora-riemannian-reproduction.git
cd moelora-riemannian-reproduction

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The ScienceQA experiment uses `meta-llama/Llama-3.2-3B`; access to the model on Hugging Face may require accepting the model's terms and authenticating locally.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The tests cover forward equivalence, exact local expert/gate gradients, Top-K routing invariants, the Top-1 control, Riemannian preconditioning, finite updates, and similarity diagnostics.

## Synthetic reproduction

Default experiment:

```bash
python experiment.py \
  --steps 300 \
  --seeds 0 1 2 \
  --device auto \
  --out results/main
```

Single configuration:

```bash
python train.py \
  --mode moe-riemannian \
  --top-k 4 \
  --steps 300 \
  --device cpu \
  --out results/single
```

Expert-similarity experiment:

```bash
python experiment.py \
  --steps 300 \
  --seeds 0 1 2 3 4 \
  --top-ks 4 8 \
  --similarity-every 20 \
  --diagnostic-size 256 \
  --device auto \
  --out results/expert_similarity
```

## Synthetic results

The committed baseline uses 3 seeds × 4 Top-K values × 2 methods × 300 steps.

| Top-K | RSGD validation MSE | gRSGD validation MSE | RSGD − gRSGD |
| ---: | ---: | ---: | ---: |
| 1 | 0.73915 | 0.73915 | 0.00000 |
| 2 | 0.57654 | 0.54514 | 0.03139 |
| 4 | 0.64738 | 0.58914 | 0.05823 |
| 8 | 0.78362 | 0.74844 | 0.03518 |

Top-1 is an important control: after renormalization, the selected gate is exactly 1, so the two backward rules coincide. For K > 1, the committed runs show lower final validation MSE for gRSGD in all nine paired runs. These results are evidence for this small synthetic setting only; they do not establish the paper's full benchmark claims.

## ScienceQA pilot

The real-task experiment injects MoE-LoRA into the `q_proj`, `k_proj`, `v_proj`, and `o_proj` projections of Llama-3.2-3B.

Example:

```bash
python real_task/train_scienceqa.py --mode riemannian --top_k 4
python real_task/train_scienceqa.py --mode moe-riemannian --top_k 4
```

Current defaults include rank 4, 20 experts, expert learning rate `3e-5`, gate learning rate `3e-8`, and a text-only ScienceQA subset.

### Important protocol differences

The real-task code should currently be treated as a **pilot reproduction**, not an exact paper benchmark reproduction.

| Area | This repository | Consequence |
| --- | --- | --- |
| Data | 1,000 train / 200 validation text-only ScienceQA examples | Smaller and simplified evaluation setting |
| Training | 100 update steps by default | Short pilot rather than full training protocol |
| Batch | Batch size 1 | Different effective optimization regime |
| Scheduler | None | Learning-rate trajectory differs |
| Router clipping | None | Gate optimization differs |
| Evaluation | Teacher-forced language-model loss | Not directly comparable with generation accuracy |
| Seeds | One real-task seed | No uncertainty estimate yet |
| Expert computation | Dense computation followed by routing mask | Reproduces routing semantics, not sparse-dispatch efficiency |

For this reason, numbers under `results/real_task/` should not be presented as direct reproductions of the paper's benchmark scores.

## Reproducibility notes

Synthetic paired comparisons use the same task, initialization, and minibatch sequence within a seed. Run metadata records configuration and environment information in the generated outputs.

For stronger real-task reproducibility, future runs should additionally record the exact model revision, dataset revision, package versions, GPU type, CUDA version, Git commit, and all training/evaluation hyperparameters.

## Limitations

- The main controlled evidence is from a small synthetic regression task.
- Only three seeds are used in the committed synthetic baseline.
- Changing Top-K also changes active capacity and routing behavior, so it does not isolate entropy as a causal variable.
- No learning-rate sweep currently separates rescaling effects from a larger effective expert step.
- The LLM experiment is intentionally compute-limited and is not yet protocol-matched to the paper.
- The real-task validation metric is loss rather than benchmark generation accuracy.
- Dense expert computation does not demonstrate the systems efficiency of sparse MoE dispatch.

## Next experiments

The most useful extensions are:

1. **Entropy × rescaling exponent:** hold K fixed and sweep router temperature and a backward factor \(g^\alpha\).
2. **Learning-rate control:** compare gRSGD against RSGD with carefully matched/tuned effective expert learning rates.
3. **Geometry diagnostics:** relate Gram-matrix conditioning, expert similarity, routing entropy, and gradient cancellation.
4. **Faithful ScienceQA protocol:** add gradient accumulation, scheduler, gate clipping, generation-based evaluation, multiple seeds, and explicit model/dataset revisions.

## Reference

- Sun et al., *A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation Models*, ICML 2025 / arXiv:2502.15828.
- Official implementation: THUDM/MoELoRA_Riemannian.

This is an independent reproduction implementation. It is not the official repository of the paper authors.
