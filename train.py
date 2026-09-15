"""Synthetic teacher regression and a paired, reproducible training run."""
import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

try:
    from .models.moe_lora import MoELoRA, MODES
    from .optim.riemannian_sgd import RiemannianSGD
except ImportError:
    from models.moe_lora import MoELoRA, MODES
    from optim.riemannian_sgd import RiemannianSGD


@dataclass
class Config:
    steps: int = 300
    batch_size: int = 64
    hidden_dim: int = 32
    output_dim: int = 8
    num_experts: int = 8
    rank: int = 4
    train_size: int = 2048
    val_size: int = 512
    lr: float = 0.1
    router_lr: float = 0.01
    damping: float = 0.01
    temperature: float = 1.0
    log_every: int = 10
    similarity_every: int = 20
    diagnostic_size: int = 256
    device: str = "auto"
    threads: int = 1


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this PyTorch installation; use --device cpu")
    return device


def make_task(cfg, seed):
    """One fixed nonlinear teacher, independent of student top_k and mode.

    Inputs come from overlapping Gaussian clusters. A smooth input-dependent
    mixture of low-rank residual maps supplies targets plus small noise.
    The student knows only the shared base W; teacher routing is never supplied.
    """
    rng = torch.Generator().manual_seed(10000 + seed)
    randn = lambda *shape: torch.randn(*shape, generator=rng)
    e, d, o, r = cfg.num_experts, cfg.hidden_dim, cfg.output_dim, cfg.rank
    centers = randn(e, d)
    base = randn(o, d) / math.sqrt(d)
    A = randn(e, r, d) / math.sqrt(d)
    B = randn(e, o, r) / math.sqrt(r)
    n = cfg.train_size + cfg.val_size
    labels = torch.randint(e, (n,), generator=rng)
    x = 0.7 * centers[labels] + randn(n, d)
    gate = (1.5 * x @ centers.T / math.sqrt(d)).softmax(-1)
    residuals = torch.einsum("ner,eor->neo", torch.einsum("nd,erd->ner", x, A), B)
    y = x @ base.T + (gate.unsqueeze(-1) * residuals).sum(1) + 0.02 * randn(n, o)
    return x[:cfg.train_size], y[:cfg.train_size], x[cfg.train_size:], y[cfg.train_size:], base


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_vector_pairwise_cosine(vectors, eps=1e-12):
    """Mean cosine over valid expert pairs for vectors shaped [E, D]."""
    norms = vectors.norm(dim=-1)
    unit = vectors / norms.clamp_min(eps).unsqueeze(-1)
    similarities = unit @ unit.T
    pair_mask = torch.triu(
        torch.ones(vectors.shape[0], vectors.shape[0], dtype=torch.bool,
                   device=vectors.device), diagonal=1)
    pair_mask &= (norms > eps).unsqueeze(0) & (norms > eps).unsqueeze(1)
    values = similarities[pair_mask]
    return ((values.mean().item() if values.numel() else float("nan")),
            int(values.numel()))


def mean_output_pairwise_cosine(expert_outputs, eps=1e-12):
    """Mean cos(E_i(x), E_j(x)) over inputs and expert pairs."""
    outputs = expert_outputs.reshape(-1, expert_outputs.shape[-2],
                                     expert_outputs.shape[-1])
    norms = outputs.norm(dim=-1)
    unit = outputs / norms.clamp_min(eps).unsqueeze(-1)
    similarities = torch.einsum("peo,pfo->pef", unit, unit)
    pair_mask = torch.triu(
        torch.ones(outputs.shape[1], outputs.shape[1], dtype=torch.bool,
                   device=outputs.device), diagonal=1).unsqueeze(0)
    pair_mask = pair_mask.expand(outputs.shape[0], -1, -1)
    pair_mask = pair_mask & ((norms > eps).unsqueeze(1)
                             & (norms > eps).unsqueeze(2))
    values = similarities[pair_mask]
    return ((values.mean().item() if values.numel() else float("nan")),
            int(values.numel()))


def similarity_diagnostics(model, x, y):
    """Measure expert redundancy/interference without touching training grads."""
    was_training = model.training
    model.eval()
    # Raw E_i(x), without gates: routing zeros must not bias K=4 versus K=8.
    output_similarity, output_values = mean_output_pairwise_cosine(
        model.expert_outputs(x))
    # B_i A_i avoids comparing the non-unique LoRA factors separately.
    update_similarity, update_pairs = mean_vector_pairwise_cosine(
        torch.bmm(model.B, model.A).flatten(start_dim=1))
    prediction, _ = model(x)
    diagnostic_loss = torch.nn.functional.mse_loss(prediction, y)
    # autograd.grad returns separate tensors and does not accumulate into .grad.
    grad_A, grad_B = torch.autograd.grad(diagnostic_loss, (model.A, model.B))
    gradient_similarity, gradient_pairs = mean_vector_pairwise_cosine(
        torch.cat((grad_A.flatten(start_dim=1), grad_B.flatten(start_dim=1)), dim=1))
    model.train(was_training)
    return dict(diagnostic_loss=diagnostic_loss.item(),
                expert_output_similarity=output_similarity,
                lora_update_similarity=update_similarity,
                gradient_similarity=gradient_similarity,
                output_valid_values=output_values,
                update_valid_pairs=update_pairs,
                gradient_valid_pairs=gradient_pairs)


def run(cfg, mode, top_k, seed, out_dir):
    if min(cfg.steps, cfg.log_every, cfg.similarity_every, cfg.diagnostic_size,
           cfg.batch_size, cfg.train_size, cfg.val_size, cfg.threads) < 1:
        raise ValueError("Steps, sizes, log interval and threads must be positive")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(seed)
    device = resolve_device(cfg.device)
    x, y, vx, vy, base = make_task(cfg, seed)
    model = MoELoRA(cfg.hidden_dim, cfg.output_dim, cfg.num_experts,
                    top_k, cfg.rank, mode, cfg.temperature).to(device)
    with torch.no_grad():
        model.base.weight.copy_(base.to(device))
    x, y, vx, vy = [t.to(device) for t in (x, y, vx, vy)]
    optimizer = RiemannianSGD(model.A, model.B, cfg.lr, cfg.damping)
    router_optimizer = torch.optim.SGD(model.router.parameters(), lr=cfg.router_lr)
    # Separate generator: same minibatches and initialization in all paired runs.
    batch_rng = torch.Generator().manual_seed(20000 + seed)
    counts = torch.zeros(cfg.num_experts, device=device, dtype=torch.long)
    logs, losses, similarity_logs = [], [], []
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with torch.no_grad():
        initial_val = torch.nn.functional.mse_loss(model(vx)[0], vy).item()
    diagnostic_x = vx[:min(cfg.diagnostic_size, len(vx))]
    diagnostic_y = vy[:min(cfg.diagnostic_size, len(vy))]
    similarity_logs.append(dict(step=0, mode=mode, top_k=top_k, seed=seed,
                                **similarity_diagnostics(model, diagnostic_x, diagnostic_y)))
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(cfg.train_size, (cfg.batch_size,), generator=batch_rng).to(device)
        optimizer.zero_grad(set_to_none=True)
        router_optimizer.zero_grad(set_to_none=True)
        pred, routing = model(x[idx])
        loss = torch.nn.functional.mse_loss(pred, y[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at {mode}, k={top_k}, step={step}")
        loss.backward()
        optimizer.step()
        router_optimizer.step()
        losses.append(loss.item())
        gates, probs, mask = [routing[key].detach() for key in ("gates", "probabilities", "mask")]
        step_counts = mask.sum(0)
        counts += step_counts
        if step == 1 or step % cfg.log_every == 0 or step == cfg.steps:
            # Batch metrics and gradients are PRE-update; validation is POST-update.
            with torch.no_grad():
                val_loss = torch.nn.functional.mse_loss(model(vx)[0], vy).item()
            row = dict(step=step, mode=mode, top_k=top_k, seed=seed,
                       train_loss=loss.item(), val_loss=val_loss,
                       mean_selected_gate=gates[mask].mean().item(),
                       gate_entropy=(-(gates * gates.clamp_min(1e-30).log()).sum(-1)).mean().item(),
                       softmax_entropy=(-(probs * probs.clamp_min(1e-30).log()).sum(-1)).mean().item())
            for i in range(cfg.num_experts):
                row[f"expert_{i}_selected_count"] = step_counts[i].item()
                row[f"expert_{i}_selected_cumulative"] = counts[i].item()
                row[f"expert_{i}_mean_gate"] = (gates[:, i].sum() / step_counts[i].clamp_min(1)).item()
                row[f"expert_{i}_grad_raw"] = optimizer.last_stats["raw"][i].item()
                row[f"expert_{i}_grad_preconditioned"] = optimizer.last_stats["preconditioned"][i].item()
            logs.append(row)
        if step % cfg.similarity_every == 0 or step == cfg.steps:
            similarity_logs.append(dict(
                step=step, mode=mode, top_k=top_k, seed=seed,
                **similarity_diagnostics(model, diagnostic_x, diagnostic_y)))
    with torch.no_grad():
        final_train = torch.nn.functional.mse_loss(model(x)[0], y).item()
    summary = dict(mode=mode, top_k=top_k, seed=seed, initial_val_loss=initial_val,
                   final_val_loss=logs[-1]["val_loss"], final_train_loss=final_train,
                   tail_train_loss=sum(losses[-50:]) / len(losses[-50:]),
                   final_gate_entropy=logs[-1]["gate_entropy"],
                   seconds=time.perf_counter() - start)
    tag = f"{mode}_k{top_k}_seed{seed}"
    write_csv(output_dir / f"{tag}.csv", logs)
    write_csv(output_dir / f"{tag}_similarity.csv", similarity_logs)
    metadata = dict(config=asdict(cfg), **summary, torch_version=torch.__version__,
                    python=platform.python_version(), device=str(device),
                    device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor())
    (output_dir / f"{tag}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{tag}: train={final_train:.5f} val={summary['final_val_loss']:.5f} ({summary['seconds']:.1f}s)", flush=True)
    return logs, summary, similarity_logs


def add_config_args(parser):
    for name, value in asdict(Config()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--mode", choices=MODES, default=MODES[0])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/single")
    args = parser.parse_args()
    run(Config(**{key: getattr(args, key) for key in asdict(Config())}),
        args.mode, args.top_k, args.seed, args.out)
