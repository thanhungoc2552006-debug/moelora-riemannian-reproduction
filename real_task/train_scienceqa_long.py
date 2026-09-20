
import argparse
import gc
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_lora import inject_moe_lora
from riemannian_sgd import RiemannianSGD
from diagnostics import compute_diagnostics, compute_update_cosine


# ============================================================
# Config
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.2-3B"

SEED = 42
MAX_LENGTH = 256

RANK = 4
NUM_EXPERTS = 20

EXPERT_LR = 3e-5
GATE_LR = 3e-8
REG = 1e-6


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# ScienceQA formatting
# ============================================================

def build_prompt(example):

    context = example.get("hint", "")

    if context is None or str(context).strip() == "":
        context = "N/A"

    question = example["question"]
    choices = example["choices"]

    options = []

    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        options.append(f"({letter}) {choice}")

    options = "\n".join(options)

    prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Options:\n{options}\n"
        f"Answer:"
    )

    return prompt


def build_target(example):

    answer_idx = int(example["answer"])
    answer_letter = chr(ord("A") + answer_idx)

    return f" The answer is {answer_letter}."


def get_answer_letter(example):
    return chr(ord("A") + int(example["answer"]))


# ============================================================
# Dataset
# ============================================================

def load_scienceqa():

    ds = load_dataset(
        "derek-thomas/ScienceQA"
    )

    # Keep current reproduction setting:
    # text-only ScienceQA
    train_ds = ds["train"].filter(
        lambda x: x["image"] is None
    )

    val_ds = ds["validation"].filter(
        lambda x: x["image"] is None
    )

    test_ds = ds["test"].filter(
        lambda x: x["image"] is None
    )

    train_ds = train_ds.shuffle(
        seed=SEED
    )

    print(
        f"Text-only train      : {len(train_ds)}"
    )
    print(
        f"Text-only validation : {len(val_ds)}"
    )
    print(
        f"Text-only test       : {len(test_ds)}"
    )

    return train_ds, val_ds, test_ds


# ============================================================
# Training tokenization
# ============================================================

def tokenize_training_dataset(
    dataset,
    tokenizer,
):

    def tokenize(example):

        prompt = build_prompt(example)
        target = build_target(example)

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        target_ids = tokenizer(
            target,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        target_ids = (
            target_ids
            + [tokenizer.eos_token_id]
        )

        input_ids = (
            prompt_ids + target_ids
        )[:MAX_LENGTH]

        prompt_len = min(
            len(prompt_ids),
            len(input_ids),
        )

        labels = (
            [-100] * prompt_len
            + input_ids[prompt_len:]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": (
                [1] * len(input_ids)
            ),
            "labels": labels,
        }

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenizing training data",
    )


def make_collate_fn(tokenizer):

    def collate(batch):

        max_len = max(
            len(x["input_ids"])
            for x in batch
        )

        input_ids = []
        attention_mask = []
        labels = []

        for x in batch:

            pad_len = (
                max_len
                - len(x["input_ids"])
            )

            input_ids.append(
                x["input_ids"]
                + [tokenizer.pad_token_id]
                * pad_len
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
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }

    return collate


# ============================================================
# Validation loss
# ============================================================

@torch.no_grad()
def evaluate_loss(
    model,
    loader,
):

    was_training = model.training
    model.eval()

    losses = []

    for batch in loader:

        batch = {
            k: v.cuda(
                non_blocking=True
            )
            for k, v in batch.items()
        }

        outputs = model(**batch)

        losses.append(
            outputs.loss.item()
        )

    if was_training:
        model.train()

    return float(
        np.mean(losses)
    )


# ============================================================
# Final QA accuracy
# ============================================================

def parse_answer(text):

    text = text.upper().strip()

    # Expected:
    # "THE ANSWER IS B."
    match = re.search(
        r"ANSWER\s+IS\s+\(?([A-E])\)?",
        text,
    )

    if match:
        return match.group(1)

    # Also allow direct generation: "B"
    match = re.match(
        r"^\s*\(?([A-E])\)?(?:[\s\.\)]|$)",
        text,
    )

    if match:
        return match.group(1)

    return None


@torch.no_grad()
def evaluate_accuracy(
    model,
    tokenizer,
    dataset,
    batch_size=8,
):

    was_training = model.training
    model.eval()

    old_padding_side = (
        tokenizer.padding_side
    )

    tokenizer.padding_side = "left"

    correct = 0
    total = 0
    invalid = 0

    for start in range(
        0,
        len(dataset),
        batch_size,
    ):

        batch_examples = dataset[
            start:
            min(
                start + batch_size,
                len(dataset),
            )
        ]

        # HF Dataset slicing returns dict of lists
        n = len(
            batch_examples["question"]
        )

        prompts = []
        gold = []

        for i in range(n):

            example = {
                key: batch_examples[key][i]
                for key
                in batch_examples.keys()
            }

            prompts.append(
                build_prompt(example)
            )

            gold.append(
                get_answer_letter(example)
            )

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        encoded = {
            k: v.cuda()
            for k, v in encoded.items()
        }

        input_length = (
            encoded["input_ids"].shape[1]
        )

        generated = model.generate(
            **encoded,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=(
                tokenizer.pad_token_id
            ),
            eos_token_id=(
                tokenizer.eos_token_id
            ),
        )

        new_tokens = generated[
            :,
            input_length:
        ]

        texts = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )

        for prediction, answer in zip(
            texts,
            gold,
        ):

            predicted_letter = (
                parse_answer(prediction)
            )

            if predicted_letter is None:
                invalid += 1

            if predicted_letter == answer:
                correct += 1

            total += 1

    tokenizer.padding_side = (
        old_padding_side
    )

    if was_training:
        model.train()

    return {
        "accuracy": (
            correct / total
            if total > 0
            else 0.0
        ),
        "correct": correct,
        "total": total,
        "invalid_predictions": invalid,
    }


# ============================================================
# Long training
# ============================================================

def run_experiment(
    mode,
    top_k,
    epochs,
    batch_size,
    grad_accum,
    eval_every,
    max_optimizer_steps=None,
):

    set_seed(SEED)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    train_raw, val_raw, test_raw = (
        load_scienceqa()
    )

    train_tok = (
        tokenize_training_dataset(
            train_raw,
            tokenizer,
        )
    )

    val_tok = (
        tokenize_training_dataset(
            val_raw,
            tokenizer,
        )
    )

    collate_fn = make_collate_fn(
        tokenizer
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_tok,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_tok,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Fixed validation batch
    # for BA / OUT / entropy diagnostics
    diagnostic_batch = next(
        iter(val_loader)
    )

    print("\nLoading Llama-3.2-3B...")

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
        )
    )

    model.cuda()

    model.config.use_cache = False

    # Inject MoE-LoRA
    injected = inject_moe_lora(
        model,
        rank=RANK,
        num_experts=NUM_EXPERTS,
        top_k=top_k,
        mode=mode,
    )

    # Robust to mutate-in-place
    # or return-model implementations
    if isinstance(
        injected,
        torch.nn.Module,
    ):
        model = injected

    # Router parameters
    gate_params = []

    for module in model.modules():

        if (
            module.__class__.__name__
            == "MoELoRALinear"
        ):

            gate_params.extend(
                module.gate.parameters()
            )

    gate_optimizer = torch.optim.SGD(
        gate_params,
        lr=GATE_LR,
    )

    expert_optimizer = RiemannianSGD(
        model,
        lr=EXPERT_LR,
        reg=REG,
    )

    gate_optimizer.zero_grad()
    expert_optimizer.zero_grad()

    # ----------------------------------------
    # Output
    # ----------------------------------------

    output_dir = Path(
        "results/real_task/long_run"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    method = (
        "grsgd"
        if mode == "moe-riemannian"
        else "rsgd"
    )

    run_name = (
        f"{method}_k{top_k}"
        f"_e{epochs}"
        f"_mb{batch_size}"
        f"_ga{grad_accum}"
    )

    csv_path = (
        output_dir
        / f"{run_name}.csv"
    )

    summary_path = (
        output_dir
        / f"{run_name}_summary.json"
    )

    history = []

    # ----------------------------------------
    # Number of steps
    # ----------------------------------------

    optimizer_steps_per_epoch = (
        math.ceil(
            len(train_loader)
            / grad_accum
        )
    )

    natural_total_steps = (
        optimizer_steps_per_epoch
        * epochs
    )

    total_optimizer_steps = (
        natural_total_steps
    )

    if max_optimizer_steps is not None:
        total_optimizer_steps = min(
            total_optimizer_steps,
            max_optimizer_steps,
        )

    print("\n==============================")
    print(f"Mode            : {mode}")
    print(f"Experts         : {NUM_EXPERTS}")
    print(f"Top-K           : {top_k}")
    print(f"Rank            : {RANK}")
    print(f"Micro batch     : {batch_size}")
    print(f"Grad accumulation: {grad_accum}")
    print(
        f"Effective batch : "
        f"{batch_size * grad_accum}"
    )
    print(f"Epochs          : {epochs}")
    print(
        f"Optimizer steps : "
        f"{total_optimizer_steps}"
    )
    print("==============================\n")

    optimizer_step = 0
    micro_step = 0

    running_loss = 0.0
    running_micro_batches = 0

    stop_training = False

    for epoch in range(epochs):

        if stop_training:
            break

        model.train()

        accum_count = 0

        for batch_idx, batch in enumerate(
            train_loader
        ):

            batch = {
                k: v.cuda(
                    non_blocking=True
                )
                for k, v
                in batch.items()
            }

            outputs = model(**batch)

            raw_loss = outputs.loss

            running_loss += (
                raw_loss.item()
            )

            running_micro_batches += 1

            # Gradient accumulation
            loss = (
                raw_loss
                / grad_accum
            )

            loss.backward()

            micro_step += 1
            accum_count += 1

            is_last_batch = (
                batch_idx
                == len(train_loader) - 1
            )

            boundary = (
                accum_count == grad_accum
                or is_last_batch
            )

            if not boundary:
                continue

            # Correct last partial
            # accumulation group
            if accum_count < grad_accum:

                correction = (
                    grad_accum
                    / accum_count
                )

                for p in model.parameters():

                    if (
                        p.requires_grad
                        and p.grad is not None
                    ):
                        p.grad.mul_(
                            correction
                        )

            optimizer_step += 1

            should_eval = (
                optimizer_step
                % eval_every
                == 0
                or optimizer_step
                == total_optimizer_steps
            )

            # IMPORTANT:
            # after backward,
            # before optimizer.step
            update_cosine = None

            if should_eval:

                update_cosine = (
                    compute_update_cosine(
                        model,
                        reg=REG,
                    )
                )

            # Actual update
            gate_optimizer.step()
            expert_optimizer.step()

            gate_optimizer.zero_grad()
            expert_optimizer.zero_grad()

            accum_count = 0

            # --------------------------------
            # Evaluation + diagnostics
            # --------------------------------

            if should_eval:

                val_loss = evaluate_loss(
                    model,
                    val_loader,
                )

                diagnostics = (
                    compute_diagnostics(
                        model,
                        diagnostic_batch,
                    )
                )

                train_loss = (
                    running_loss
                    / max(
                        running_micro_batches,
                        1,
                    )
                )

                row = {
                    "epoch": epoch + 1,
                    "optimizer_step": (
                        optimizer_step
                    ),
                    "micro_step": (
                        micro_step
                    ),
                    "train_loss": (
                        train_loss
                    ),
                    "val_loss": (
                        val_loss
                    ),
                    "update_cosine": (
                        update_cosine
                    ),
                    **diagnostics,
                }

                history.append(row)

                # Save immediately
                # in case Colab disconnects
                pd.DataFrame(
                    history
                ).to_csv(
                    csv_path,
                    index=False,
                )

                print(
                    f"epoch={epoch+1} | "
                    f"step={optimizer_step:04d} | "
                    f"train={train_loss:.4f} | "
                    f"val={val_loss:.4f} | "
                    f"UPD={update_cosine:.4f} | "
                    f"BA="
                    f"{diagnostics['ba_cosine']:.4f} | "
                    f"OUT="
                    f"{diagnostics['output_cosine']:.4f} | "
                    f"H="
                    f"{diagnostics['gate_entropy_norm']:.4f}"
                )

                running_loss = 0.0
                running_micro_batches = 0

            # Smoke-test cutoff
            if (
                max_optimizer_steps
                is not None
                and optimizer_step
                >= max_optimizer_steps
            ):

                stop_training = True
                break

    # ========================================================
    # Final accuracy
    # Skip for smoke test
    # ========================================================

    summary = {
        "model": MODEL_NAME,
        "mode": mode,
        "num_experts": NUM_EXPERTS,
        "top_k": top_k,
        "rank": RANK,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": (
            batch_size * grad_accum
        ),
        "optimizer_steps": (
            optimizer_step
        ),
    }

    if max_optimizer_steps is None:

        print(
            "\nRunning final "
            "ScienceQA test accuracy..."
        )

        accuracy_result = (
            evaluate_accuracy(
                model,
                tokenizer,
                test_raw,
                batch_size=max(
                    batch_size,
                    1,
                ),
            )
        )

        summary.update(
            accuracy_result
        )

        print(
            f"Test accuracy: "
            f"{accuracy_result['accuracy']:.4f} "
            f"("
            f"{accuracy_result['correct']}/"
            f"{accuracy_result['total']}"
            f")"
        )

        print(
            "Invalid generations:",
            accuracy_result[
                "invalid_predictions"
            ],
        )

    else:

        print(
            "\nSmoke test: "
            "skipping full test accuracy."
        )

    with open(
        summary_path,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"\nSaved history : {csv_path}"
    )

    print(
        f"Saved summary : {summary_path}"
    )

    del model

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "riemannian",
            "moe-riemannian",
        ],
        required=True,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--max_optimizer_steps",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    run_experiment(
        mode=args.mode,
        top_k=args.top_k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        eval_every=args.eval_every,
        max_optimizer_steps=(
            args.max_optimizer_steps
        ),
    )
