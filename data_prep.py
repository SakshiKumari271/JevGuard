import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Model Backbone chunna
MODEL_NAME = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer load ho gaya!")

# 2. Dataset load karna (Direct parquet formatted MNLI)
# 'nyu-mll/glue' namespace-compliant path hai
print("Dataset download ho raha hai...")
raw_dataset = load_dataset("nyu-mll/glue", "mnli")

# Fast experimentation ke liye subsets
train_subset = raw_dataset["train"].shuffle(seed=42).select(range(10000))
eval_subset = raw_dataset["validation_matched"].shuffle(seed=42).select(range(1000))


# 3. Tokenization function
def preprocess_function(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )


print("Data tokenize ho raha hai...")
tokenized_train = train_subset.map(
    preprocess_function, batched=True, batch_size=500
)
tokenized_eval = eval_subset.map(
    preprocess_function, batched=True, batch_size=500
)

# PyTorch format set karna
tokenized_train.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids", "label"],
)
tokenized_eval.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids", "label"],
)

print(
    f"\nDataset Tayyar Hai!\nTrain Samples: {len(tokenized_train)} | Eval Samples: {len(tokenized_eval)}"
)

# Sanity check
sample = tokenized_train[0]
print("\n--- Tensor Verification ---")
print("Input IDs shape:", sample["input_ids"].shape)
print("Attention Mask shape:", sample["attention_mask"].shape)
print("Sample Label:", sample["label"].item())