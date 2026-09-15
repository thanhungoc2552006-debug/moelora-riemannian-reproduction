"""Compare RSGD and gRSGD over Top-K with paired seeds and plots."""
import argparse
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

try:
    from .models.moe_lora import MODES, weight_expert
    from .train import Config, add_config_args, run, write_csv
except ImportError:
    from models.moe_lora import MODES, weight_expert
    from train import Config, add_config_args, run, write_csv

LABELS = {"riemannian": "RSGD", "moe-riemannian": "gRSGD"}


def mean_sd(values):
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd


def finite_mean_sd(values):
    values = [value for value in values if math.isfinite(value)]
    return mean_sd(values) if values else (float("nan"), float("nan"))


def gradient_probe(out):
    # Controlled local derivative experiment: fixed expert output and upstream
    # gradient, varying only g. Training scatter alone confounds routing, load,
    # residuals, and cancellations across examples.
    rows = []
    for g in torch.logspace(-3, 0, 40, dtype=torch.float64):
        row = {"gate": g.item()}
        for mode in MODES:
            expert = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
            weight_expert(g, expert, mode).sum().backward()
            row[LABELS[mode]] = expert.grad.norm().item() / math.sqrt(2)
        rows.append(row)
    write_csv(out / "gradient_probe.csv", rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode in MODES:
        label = LABELS[mode]
        ax.loglog([r["gate"] for r in rows], [r[label] for r in rows], label=label)
    ax.set(xlabel="Gate g (fixed upstream gradient)", ylabel="Expert gradient / ungated gradient",
           title="Controlled backward probe: g versus sqrt(g)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "gradient_probe.png", dpi=160)
    plt.close(fig)


def plot_results(runs, summaries, top_ks, seeds, out):
    fig, axes = plt.subplots(math.ceil(len(top_ks) / 2), 2, figsize=(11, 3.4 * math.ceil(len(top_ks) / 2)), squeeze=False)
    for ax, k in zip(axes.flat, top_ks):
        for mode in MODES:
            series = [runs[(mode, k, seed)] for seed in seeds]
            steps = [row["step"] for row in series[0]]
            stats = [mean_sd([s[j]["train_loss"] for s in series]) for j in range(len(steps))]
            means, sds = zip(*stats)
            line, = ax.plot(steps, means, label=LABELS[mode])
            ax.fill_between(steps, [m-s for m,s in stats], [m+s for m,s in stats], color=line.get_color(), alpha=0.15)
        ax.set(title=f"Top-K = {k}", xlabel="Training step", ylabel="Minibatch MSE")
        ax.legend()
    for ax in list(axes.flat)[len(top_ks):]:
        ax.set_visible(False)
    fig.suptitle("Training loss: paired-seed mean ± sample SD", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "training_loss.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    lookup = {(s["mode"], s["top_k"], s["seed"]): s for s in summaries}
    aggregate = []
    for k in top_ks:
        rs = [lookup[(MODES[0], k, seed)] for seed in seeds]
        gs = [lookup[(MODES[1], k, seed)] for seed in seeds]
        gaps = [r["final_val_loss"] - g["final_val_loss"] for r,g in zip(rs, gs)]
        rel = [100 * gap / r["final_val_loss"] for gap,r in zip(gaps, rs)]
        gap_mean, gap_sd = mean_sd(gaps)
        row = dict(top_k=k, seeds=len(seeds), rsgd_val_mean=mean_sd([r["final_val_loss"] for r in rs])[0],
                   grsgd_val_mean=mean_sd([g["final_val_loss"] for g in gs])[0],
                   val_gap_mean=gap_mean, val_gap_sd=gap_sd, relative_improvement_pct=mean_sd(rel)[0],
                   train_gap_mean=mean_sd([r["final_train_loss"]-g["final_train_loss"] for r,g in zip(rs,gs)])[0],
                   rsgd_entropy_mean=mean_sd([r["final_gate_entropy"] for r in rs])[0],
                   grsgd_entropy_mean=mean_sd([g["final_gate_entropy"] for g in gs])[0])
        aggregate.append(row)
    write_csv(out / "aggregate.csv", aggregate)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(top_ks, [r["val_gap_mean"] for r in aggregate],
                     yerr=[r["val_gap_sd"] for r in aggregate], marker="o", capsize=4, label="Validation gap ± SD")
    axes[0].plot(top_ks, [r["train_gap_mean"] for r in aggregate], marker="s", label="Full training-set gap")
    axes[0].axhline(0, color="gray", linewidth=1)
    axes[0].set(xlabel="Top-K", ylabel="MSE(RSGD) − MSE(gRSGD)", title="Positive gap favors gRSGD", xticks=top_ks)
    axes[0].legend()
    for mode, key in zip(MODES, ("rsgd_entropy_mean", "grsgd_entropy_mean")):
        axes[1].plot(top_ks, [r[key] for r in aggregate], marker="o", label=LABELS[mode])
    axes[1].plot(top_ks, [math.log(k) for k in top_ks], "k--", label="Maximum log(K)")
    axes[1].set(xlabel="Top-K", ylabel="Post-Top-K entropy (nats)", title="Measure actual gate dispersion", xticks=top_ks)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "top_k_gap.png", dpi=160)
    plt.close(fig)
    return aggregate


def plot_similarities(similarity_runs, top_ks, seeds, out):
    metrics = (("expert_output_similarity", "Expert output cosine"),
               ("lora_update_similarity", "LoRA update BA cosine"),
               ("gradient_similarity", "Raw gradient cosine"))
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True, sharey="row")
    for column, mode in enumerate(MODES):
        for row_index, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_index, column]
            for k in top_ks:
                series = [similarity_runs[(mode, k, seed)] for seed in seeds]
                steps = [row["step"] for row in series[0]]
                stats = [finite_mean_sd([run[index][metric] for run in series])
                         for index in range(len(steps))]
                means, sds = zip(*stats)
                line, = ax.plot(steps, means, marker="o", markersize=3,
                                label=f"K={k}")
                ax.fill_between(steps, [m-s for m, s in stats],
                                [m+s for m, s in stats],
                                color=line.get_color(), alpha=0.15)
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title(LABELS[mode])
            if row_index == 2:
                ax.set_xlabel("Training step")
            ax.legend()
    fig.suptitle("Mean pairwise expert similarity on a fixed validation batch")
    fig.tight_layout()
    fig.savefig(out / "expert_similarity.png", dpi=160)
    plt.close(fig)


def similarity_tail_summary(similarity_runs, top_ks, seeds):
    """Summarize paired K=8 minus K=4 over the last five checkpoints."""
    if 4 not in top_ks or 8 not in top_ks:
        return []
    metrics = ("expert_output_similarity", "lora_update_similarity",
               "gradient_similarity")
    rows = []
    for mode in MODES:
        for metric in metrics:
            k4_values, k8_values, deltas = [], [], []
            for seed in seeds:
                tails = {}
                for k in (4, 8):
                    values = [row[metric] for row in similarity_runs[(mode, k, seed)]
                              if math.isfinite(row[metric])][-5:]
                    tails[k] = sum(values) / len(values)
                k4_values.append(tails[4])
                k8_values.append(tails[8])
                deltas.append(tails[8] - tails[4])
            delta_mean, delta_sd = mean_sd(deltas)
            rows.append(dict(mode=LABELS[mode], metric=metric,
                             k4_tail_mean=mean_sd(k4_values)[0],
                             k8_tail_mean=mean_sd(k8_values)[0],
                             paired_k8_minus_k4_mean=delta_mean,
                             paired_k8_minus_k4_sd=delta_sd,
                             seeds=len(seeds)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=Path("results/main"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.top_ks)) != len(args.top_ks):
        parser.error("Seeds and top-ks must not contain duplicates")
    if any(k < 1 or k > args.num_experts for k in args.top_ks):
        parser.error("Each top-k must be between 1 and num-experts")
    cfg = Config(**{key: getattr(args, key) for key in asdict(Config())})
    runs, summaries, similarity_runs = {}, [], {}
    for seed in args.seeds:
        for k in args.top_ks:
            for mode in MODES:
                logs, summary, similarity_logs = run(cfg, mode, k, seed, args.out)
                runs[(mode, k, seed)] = logs
                similarity_runs[(mode, k, seed)] = similarity_logs
                summaries.append(summary)
    write_csv(args.out / "summary.csv", summaries)
    aggregate = plot_results(runs, summaries, args.top_ks, args.seeds, args.out)
    write_csv(args.out / "similarity.csv",
              [row for run_rows in similarity_runs.values() for row in run_rows])
    plot_similarities(similarity_runs, args.top_ks, args.seeds, args.out)
    tail_rows = similarity_tail_summary(similarity_runs, args.top_ks, args.seeds)
    if tail_rows:
        write_csv(args.out / "similarity_tail_summary.csv", tail_rows)
    gradient_probe(args.out)
    for row in aggregate:
        print(f"K={row['top_k']}: paired validation gap={row['val_gap_mean']:.6f} ± {row['val_gap_sd']:.6f}; improvement={row['relative_improvement_pct']:.2f}%")
    for row in tail_rows:
        print(f"{row['mode']} {row['metric']}: K8-K4="
              f"{row['paired_k8_minus_k4_mean']:.4f} ± "
              f"{row['paired_k8_minus_k4_sd']:.4f}")
    print(f"Saved CSV, metadata and figures to {args.out.resolve()}")


if __name__ == "__main__":
    main()
