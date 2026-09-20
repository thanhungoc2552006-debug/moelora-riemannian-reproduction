
import gc
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from prepare_scienceqa import load_scienceqa
from moe_lora import inject_moe_lora
from riemannian_sgd import RiemannianSGD
from diagnostics import compute_diagnostics, compute_update_cosine


MODEL_NAME = "meta-llama/Llama-3.2-3B"

SEED = 42
MAX_LENGTH = 256

RANK = 4
NUM_EXPERTS = 20

EXPERT_LR = 3e-5
GATE_LR = 3e-8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_dataset(dataset, tokenizer):

    def tokenize_example(example):
        prompt_ids = tokenizer(
            example["prompt"],
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"]

        target_ids = tokenizer(
            example["target"],
            add_special_tokens=False,
            truncation=True,
            max_length=32,
        )["input_ids"]

        input_ids = (prompt_ids + target_ids)[:MAX_LENGTH]

        labels = (
            [-100] * len(prompt_ids)
            + target_ids
        )[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
    )


def make_collate_fn(tokenizer):

    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for x in batch:
            pad_len = max_len - len(x["input_ids"])

            input_ids.append(
                x["input_ids"]
                + [tokenizer.pad_token_id] * pad_len
            )

            attention_mask.append(
                x["attention_mask"]
                + [0] * pad_len
            )

            labels.append(
                x["labels"]
                + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    return collate_fn


@torch.no_grad()
def evaluate(model, loader, max_batches=100):
    model.eval()

    losses = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        batch = {
            k: v.to(model.device)
            for k, v in batch.items()
        }

        loss = model(**batch).loss
        losses.append(loss.item())

    model.train()

    return float(np.mean(losses))


def run_experiment(
    mode,
    top_k,
    train_tok,
    val_tok,
    tokenizer,
    num_steps=100,
    eval_every=20,
):

    set_seed(SEED)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.config.use_cache = False

    set_seed(SEED)

    inject_moe_lora(
        model,
        rank=RANK,
        num_experts=NUM_EXPERTS,
        top_k=top_k,
        alpha=8.0,
        mode=mode,
    )

    collate_fn = make_collate_fn(tokenizer)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_tok,
        batch_size=1,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_tok,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Same validation sample is used for diagnostics
    diagnostic_batch = next(iter(val_loader))

    expert_optimizer = RiemannianSGD(
        model,
        lr=EXPERT_LR,
        reg=1e-6,
    )

    gate_params = [
        p for name, p in model.named_parameters()
        if ".gate." in name and p.requires_grad
    ]

    gate_optimizer = torch.optim.SGD(
        gate_params,
        lr=GATE_LR,
    )

    train_losses = []
    val_history = []

    model.train()

    for step, batch in enumerate(train_loader):

        if step >= num_steps:
            break

        batch = {
            k: v.to(model.device)
            for k, v in batch.items()
        }

        expert_optimizer.zero_grad()
        gate_optimizer.zero_grad(set_to_none=True)

        loss = model(**batch).loss
        loss.backward()

        update_cosine = None
        if (step + 1) % eval_every == 0:
            update_cosine = compute_update_cosine(
                model,
                reg=1e-6,
            )

        expert_optimizer.step()
        gate_optimizer.step()

        train_losses.append(loss.item())

        if (step + 1) % eval_every == 0:
            avg_train = np.mean(train_losses[-eval_every:])
            val_loss = evaluate(model, val_loader)

            diagnostics = compute_diagnostics(
                model,
                diagnostic_batch,
            )

            val_history.append({
                "step": step + 1,
                "train_loss": float(avg_train),
                "val_loss": val_loss,
                "update_cosine": update_cosine,
                **diagnostics,
            })

            print(
                f"{mode:15s} | "
                f"K={top_k} | "
                f"step {step+1:03d} | "
                f"train={avg_train:.4f} | "
                f"val={val_loss:.4f} | "
                f"BA={diagnostics['ba_cosine']:.3f} | "
                f"OUT={diagnostics['output_cosine']:.3f} | "
                f"UPD={update_cosine:.3f} | "
                f"H={diagnostics['gate_entropy_norm']:.3f}"
            )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return train_losses, val_history


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["riemannian", "moe-riemannian"],
        required=True,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds, val_ds = load_scienceqa(
        train_size=1000,
        val_size=200,
        seed=SEED,
    )

    train_tok = tokenize_dataset(train_ds, tokenizer)
    val_tok = tokenize_dataset(val_ds, tokenizer)

    train_losses, val_history = run_experiment(
        mode=args.mode,
        top_k=args.top_k,
        train_tok=train_tok,
        val_tok=val_tok,
        tokenizer=tokenizer,
    )

    output_dir = Path("results/real_task/topk")
    output_dir.mkdir(parents=True, exist_ok=True)

    method = (
        "grsgd"
        if args.mode == "moe-riemannian"
        else "rsgd"
    )

    output_path = output_dir / f"{method}_k{args.top_k}.csv"

    pd.DataFrame(val_history).to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved result to: {output_path}")
