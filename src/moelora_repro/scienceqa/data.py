"""Shared seed, answer parsing and padding helpers for ScienceQA."""
import random
import re

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
