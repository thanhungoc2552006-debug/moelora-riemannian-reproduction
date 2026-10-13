"""Rebuild the comparison and validation figure from archived run records."""
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def summarize(results):
    rows, histories = [], {}
    for method, folder, paper in (
        ("RSGD", "rsgd_seed42", 62.8), ("gRSGD", "grsgd_seed42", 70.1)
    ):
        run = results / "runs" / folder
        summary = json.loads((run / "summary.json").read_text())
        assert summary["total"] > 0
        assert abs(summary["accuracy"] - summary["correct"] / summary["total"]) < 1e-12
        history = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
                   if line.strip()]
        history.sort(key=lambda row: row["step"])
        assert history and len({row["step"] for row in history}) == len(history)
        assert history[-1]["step"] == summary["optimizer_steps"]
        histories[method] = history
        rows.append(dict(method=method, seed=summary["config"]["seed"],
                         optimizer_steps=summary["optimizer_steps"], paper_accuracy_pct=paper,
                         test_accuracy_pct=100 * summary["accuracy"], correct=summary["correct"],
                         total=summary["total"], difference_from_paper_pp=100*summary["accuracy"]-paper,
                         final_validation_loss=history[-1]["validation_loss"]))
    return rows, histories


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/scienceqa")
    args = parser.parse_args(argv)
    rows, histories = summarize(args.results)
    with (args.results / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, history in histories.items():
        ax.plot([r["step"] for r in history], [r["validation_loss"] for r in history],
                marker="o", linewidth=2, label=method)
    ax.set(xlabel="Optimizer update", ylabel="Full validation answer-token loss",
           title="ScienceQA: recorded validation checkpoints (seed 42)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.text(0.5, 0.015,
             "Missing records: RSGD 100/200; gRSGD 300. Lines connect recorded points.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = args.results / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "validation_loss.png", dpi=160)
    plt.close(fig)
    print(f"Rebuilt comparison.csv and figures/validation_loss.png in {args.results}")


if __name__ == "__main__":
    main()
