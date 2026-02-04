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
    precision_recall_fscore_support,
    classification_report, hamming_loss
)

from normalizer import normalize

# ======================
# CONFIG
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 8
MAX_LEN = 192
SEED = 42
THRESHOLD = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)

# ======================
# DATA (HUGGINGFACE)
# ======================
def clean_bangla(text):
    text = normalize(str(text))
    text = re.sub(r"http\S+", "[URL]", text)
    return text.strip()

from datasets import load_dataset
ds = load_dataset("aplycaebous/BanTH")

LABEL_COLS = [
    "Political", "Religious", "Gender",
    "Personal Offense", "Abusive/Violence",
    "Origin", "Body Shaming", "Misc"
]

NUM_CLASSES = len(LABEL_COLS)

def prepare_split(split):
    texts = [clean_bangla(x) for x in split["Text"]]
    labels = np.stack([split[c] for c in LABEL_COLS], axis=1).astype("float32")
    return texts, labels

train_texts, train_labels = prepare_split(ds["train"])
dev_texts, dev_labels     = prepare_split(ds["validation"])
test_texts, test_labels   = prepare_split(ds["test"])

print("Labels:", LABEL_COLS)

# ======================
# DATASET
# ======================
class HateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
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
# TRAIN + EVAL
# ======================
def train_and_eval(name, model_name):
    print(f"\n\n================= {name} =================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Raw text example:")
    print(train_texts[0])
    
    print("\nTokenized (first 30 tokens):")
    print(tokenizer.tokenize(train_texts[0])[:30])

    train_loader = DataLoader(
        HateDataset(train_texts, train_labels, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    dev_loader = DataLoader(
        HateDataset(dev_texts, dev_labels, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_loader = DataLoader(
        HateDataset(test_texts, test_labels, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES
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

        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for batch in dev_loader:
                out = model(
                    input_ids=batch["ids"].to(DEVICE),
                    attention_mask=batch["mask"].to(DEVICE)
                )
                probs = torch.sigmoid(out.logits)
                preds.append((probs > THRESHOLD).cpu().numpy())
                gold.append(batch["labels"].numpy())

            if ep == 0:
                print(
                    "Logits stats | "
                    f"min={out.logits.min().item():.3f}, "
                    f"mean={out.logits.mean().item():.3f}, "
                    f"max={out.logits.max().item():.3f}"
                )
                print(
                    "Probs stats  | "
                    f"min={probs.min().item():.3f}, "
                    f"mean={probs.mean().item():.3f}, "
                    f"max={probs.max().item():.3f}"
                )

        preds = np.vstack(preds)
        gold = np.vstack(gold)

        _, _, dev_macro_f, _ = precision_recall_fscore_support(
            gold, preds, average="macro", zero_division=0
        )
        print("DEV Hamming Loss:", hamming_loss(gold, preds))
        print(
            f"Epoch {ep+1}/{EPOCHS} | "
            f"Loss={total_loss/len(train_loader):.4f} | "
            f"Dev MacroF1={dev_macro_f:.4f}"
        )

        if dev_macro_f > best_dev_f1:
            best_dev_f1 = dev_macro_f
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                input_ids=batch["ids"].to(DEVICE),
                attention_mask=batch["mask"].to(DEVICE)
            )
            probs = torch.sigmoid(out.logits)
            preds.append((probs > THRESHOLD).cpu().numpy())
            gold.append(batch["labels"].numpy())

    preds = np.vstack(preds)
    gold = np.vstack(gold)

    print("Test Hamming Loss:", hamming_loss(gold, preds))

    print("\n===== PER LABEL REPORT =====")
    print(classification_report(
        gold, preds,
        target_names=LABEL_COLS,
        digits=4,
        zero_division=0
    ))

# ======================
# MODELS
# ======================
models = [
    ("BanglaBERT LARGE", "csebuetnlp/banglabert_large"),
    ("BanglaBERT (SagorSarker)", "sagorsarker/bangla-bert-base"),
    ("BanglaBERT", "csebuetnlp/banglabert"),
    ("BanglaBERT Small", "csebuetnlp/banglabert_small"),
    ("DistilBERT", "distilbert-base-multilingual-cased"),
    ("mBERT", "bert-base-multilingual-cased"),
    ("XLM-R Base", "xlm-roberta-base")
]

for n, m in models:
    train_and_eval(n, m)
