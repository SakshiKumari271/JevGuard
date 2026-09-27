import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from datasets import load_dataset
from model import JevGuardClassifier
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# 1. Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# 2. Tokenizer
MODEL_NAME = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Dataset load kiya ja raha hai...")
raw_dataset = load_dataset("nyu-mll/glue", "mnli")

# CPU-friendly fast run: 1,000 train aur 200 eval samples
train_data = raw_dataset["train"].shuffle(seed=42).select(range(1000))
eval_data = raw_dataset["validation_matched"].shuffle(seed=42).select(range(200))


def tokenize_fn(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=128,  # 128 max length CPU par 4x fast chalegi
        padding="max_length",
    )


print("Preprocessing data...")
train_tok = train_data.map(tokenize_fn, batched=True, batch_size=250)
eval_tok = eval_data.map(tokenize_fn, batched=True, batch_size=250)

train_tok.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids", "label"],
)
eval_tok.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids", "label"],
)

BATCH_SIZE = 16
train_loader = DataLoader(train_tok, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_tok, batch_size=BATCH_SIZE)

# 3. Model initialization
model = JevGuardClassifier(model_name=MODEL_NAME).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

EPOCHS = 1  # Pehle 1 epoch run karke verify karenge
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

# 4. Training Loop
print(
    f"\n--- JevGuard Training Shuru ({len(train_loader)} batches total) ---"
)
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1} - Average Loss: {avg_train_loss:.4f}")

    # 5. Fast Evaluation Loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = (correct / total) * 100
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# 6. Save Model
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/jevguard_weights.pt")
print("\nSuccess! Weights saved in 'weights/jevguard_weights.pt'")