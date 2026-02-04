import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    hamming_loss
)
from datasets import load_dataset

from normalizer import normalize

# ======================
# CONFIG
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 6
MAX_LEN = 192
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ======================
# MULTI-LABEL SETUP
# ======================
LABEL_COLS = [
    "Political", "Religious", "Gender", "Personal Offense",
    "Abusive/Violence", "Origin", "Body Shaming", "Misc"
]
NUM_CLASSES = len(LABEL_COLS)

# ======================
# DATA LOADING (HF)
# ======================
def clean_bangla(text):
    text = normalize(str(text))
    text = re.sub(r"http\S+", "[URL]", text)
    return text.strip()

print("Loading HuggingFace dataset...")
ds = load_dataset("aplycaebous/BanTH")

def process_split(split):
    df = split.to_pandas()
    df = df.dropna(subset=["Text"])

    # ✅ KEEP ONLY HATE SAMPLES (CRITICAL)
    df = df[df["Label"] == 1]

    df["text"] = df["Text"].astype(str).apply(clean_bangla)
    df["y"] = df[LABEL_COLS].values.tolist()
    return df.reset_index(drop=True)

train_df = process_split(ds["train"])
dev_df = process_split(ds["validation"])
test_df = process_split(ds["test"])

labels = LABEL_COLS
print("Labels:", labels)

# ======================
# DATASET
# ======================
class HateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "ids": encoded["input_ids"].squeeze(0),
            "mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

# ======================
# FLOPs + PARAMS
# ======================
try:
    from thop import profile
    USE_THOP = True
except:
    USE_THOP = False


def compute_stats(model):
    params = sum(p.numel() for p in model.parameters())

    if not USE_THOP:
        return params, None

    dummy_ids = torch.randint(0, 2000, (1, MAX_LEN)).to(DEVICE)
    dummy_mask = torch.ones_like(dummy_ids).to(DEVICE)

    flops, _ = profile(
        model,
        inputs=(dummy_ids, dummy_mask),
        verbose=False
    )
    return params, flops / 1e9

# ======================
# TRAIN + EVAL
# ======================
def train_and_eval(name, model_name):
    print(f"\n\n================= {name} =================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader = DataLoader(
        HateDataset(train_df.text, train_df.y, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    dev_loader = DataLoader(
        HateDataset(dev_df.text, dev_df.y, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_loader = DataLoader(
        HateDataset(test_df.text, test_df.y, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        problem_type="multi_label_classification"
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps
    )

    loss_fn = nn.BCEWithLogitsLoss()

    best_dev_f1 = -1
    best_state = None

    for ep in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            ids = batch["ids"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # ======================
        # DEV EVAL
        # ======================
        model.eval()
        preds, gold = [], []

        with torch.no_grad():
            for batch in dev_loader:
                logits = model(
                    input_ids=batch["ids"].to(DEVICE),
                    attention_mask=batch["mask"].to(DEVICE)
                ).logits

                p = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
                preds.append(p)
                gold.append(batch["labels"].numpy())

        preds = np.vstack(preds)
        gold = np.vstack(gold)

        dev_f1 = f1_score(gold, preds, average="macro", zero_division=0)
        dev_subset_acc = accuracy_score(gold, preds)
        dev_hamming = hamming_loss(gold, preds)

        print(
            f"Epoch {ep+1}/{EPOCHS} | Loss={total_loss/len(train_loader):.4f} "
            f"| Dev F1(Multi)={dev_f1:.4f} "
            f"| Subset Acc={dev_subset_acc:.4f} "
            f"| Hamming={dev_hamming:.4f}"
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    # ======================
    # TEST EVAL
    # ======================
    model.eval()
    preds, gold = [], []

    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                input_ids=batch["ids"].to(DEVICE),
                attention_mask=batch["mask"].to(DEVICE)
            ).logits

            p = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            preds.append(p)
            gold.append(batch["labels"].numpy())

    preds = np.vstack(preds)
    gold = np.vstack(gold)

    f1 = f1_score(gold, preds, average="macro", zero_division=0)
    subset_acc = accuracy_score(gold, preds)
    hamming = hamming_loss(gold, preds)

    print("\n===== TEST METRICS =====")
    print("F1 (Multi-Label):", f1)
    print("Subset Accuracy :", subset_acc)
    print("Hamming Loss    :", hamming * 100)

    params, flops = compute_stats(model)

    print("\n===== MODEL INFO =====")
    print("Architecture:", model_name)
    print("Params:", params)
    if flops:
        print("FLOPs (GF):", flops)

# ======================
# MODELS
# ======================
models = [
    ("mBERT", "bert-base-multilingual-cased"),
    ("BanglaBERT Small", "csebuetnlp/banglabert_small"),
    ("BanglaBERT", "csebuetnlp/banglabert"),
    ("BanglaBERT LARGE", "csebuetnlp/banglabert_large"),
    ("DistilBERT", "distilbert-base-multilingual-cased"),
    ("BanglaBERT (SagorSarker)", "sagorsarker/bangla-bert-base"),
    ("XLM-R Base", "xlm-roberta-base"),
    ("Bangla_HateBERT", "saroarj/BanglaHateBert")
]

for n, m in models:
    train_and_eval(n, m)