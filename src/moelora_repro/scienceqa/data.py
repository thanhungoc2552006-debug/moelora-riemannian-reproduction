

LETTERS = ["A", "B", "C", "D", "E"]


def format_example(example):
    choices = "\n".join(
        f"({LETTERS[i]}) {choice}"
        for i, choice in enumerate(example["choices"])
    )

    context = example["hint"] or ""

    prompt = f"""Context: {context}
Question: {example["question"]}
Options:
{choices}
Answer:"""

    answer = f" The answer is {LETTERS[example['answer']]}."

    return {
        "prompt": prompt,
        "target": answer,
        "text": prompt + answer,
    }


def load_scienceqa(
    train_size=1000,
    val_size=200,
    seed=42,
):
    from datasets import load_dataset

    ds = load_dataset("derek-thomas/ScienceQA")

    train_ds = (
        ds["train"]
        .filter(lambda x: x["image"] is None)
        .shuffle(seed=seed)
        .select(range(train_size))
        .map(format_example)
    )

    val_ds = (
        ds["validation"]
        .filter(lambda x: x["image"] is None)
        .shuffle(seed=seed)
        .select(range(val_size))
        .map(format_example)
    )

    return train_ds, val_ds
