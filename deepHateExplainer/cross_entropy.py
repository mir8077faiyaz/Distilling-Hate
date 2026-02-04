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
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from sklearn.model_selection import train_test_split

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
# DATA LOADING
# ======================
def clean_bangla(text):
    text = normalize(str(text))
    text = re.sub(r"http\S+", "[URL]", text)
    return text.strip()

def load_fixed_splits(train_path, val_path, test_path):
    def process_df(path):
        df = pd.read_csv(path)
        # MODIFIED: Drop rows if either 'text' OR 'label' is missing
        df = df.dropna(subset=["text", "label"])
        df["label"] = df["label"].astype(str).str.strip()
        df["text"] = df["text"].apply(clean_bangla)
        return df.reset_index(drop=True)

    print(f"Loading provided splits...")
    return (
        process_df(train_path),
        process_df(val_path),
        process_df(test_path)
    )

# Use the specific file names from the repo
train_df, dev_df, test_df = load_fixed_splits(
    "train.csv", "validation.csv", "test.csv"
)

# The rest of your label mapping remains the same
labels = sorted(train_df.label.unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
NUM_CLASSES = len(labels)

train_df["y"] = train_df.label.map(label2id)
dev_df["y"] = dev_df.label.map(label2id)
test_df["y"] = test_df.label.map(label2id)



print("Labels:", labels)
print(train_df["label"].value_counts())


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
            "labels": torch.tensor(self.labels[idx])
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
        num_labels=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps
    )

    ce = nn.CrossEntropyLoss()

    best_dev_score = -1
    best_state = None

    for ep in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            ids = batch["ids"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            out = model(input_ids=ids, attention_mask=mask)
            loss = ce(out.logits, y)

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
                preds.extend(out.logits.argmax(1).cpu().numpy())
                gold.extend(batch["labels"].numpy())

        gold = np.array(gold)
        preds = np.array(preds)

        dev_acc = accuracy_score(gold, preds)
        _, _, dev_macro_f, _ = precision_recall_fscore_support(
            gold, preds, average="macro"
        )
        _, _, dev_weighted_f, _ = precision_recall_fscore_support(
            gold, preds, average="weighted"
        )

        # dev_score = (dev_acc + dev_macro_f + dev_weighted_f) / 3

        print(
            f"Epoch {ep+1}/{EPOCHS} | Loss={total_loss/len(train_loader):.4f} "
            f"| Dev Acc={dev_acc:.4f} | Dev MacroF1={dev_macro_f:.4f} "
            f"| Dev WeightedF1={dev_weighted_f:.4f}"
        )

        if dev_macro_f > best_dev_score:
            best_dev_score = dev_macro_f
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    def get_preds(loader):
        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for batch in loader:
                out = model(
                    input_ids=batch["ids"].to(DEVICE),
                    attention_mask=batch["mask"].to(DEVICE)
                )
                preds.extend(out.logits.argmax(1).cpu().numpy())
                gold.extend(batch["labels"].numpy())
        return np.array(gold), np.array(preds)

    gold, pred = get_preds(test_loader)

    acc = accuracy_score(gold, pred)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        gold, pred, average="macro"
    )
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        gold, pred, average="micro"
    )
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        gold, pred, average="weighted"
    )

    print("\n===== SUMMARY METRICS =====")
    print("Accuracy :", acc)
    print("Macro Precision :", macro_p)
    print("Macro Recall :", macro_r)
    print("Macro F1 :", macro_f)
    print("Micro Precision :", micro_p)
    print("Micro Recall :", micro_r)
    print("Micro F1 :", micro_f)
    print("Weighted Precision :", weighted_p)
    print("Weighted Recall :", weighted_r)
    print("Weighted F1 :", weighted_f)

    print("\n===== PER CLASS REPORT =====")
    print(classification_report(
        gold, pred,
        target_names=labels,
        digits=4
    ))

    params, flops = compute_stats(model)

    print("\n===== MODEL INFO =====")
    print("Architecture:", model_name)
    print("Params:", params)
    if flops:
        print("FLOPs (GF):", flops)
    else:
        print("FLOPs: install thop to enable")

# ======================
# MODELS
# ======================
models = [
    #("BanglaBERT (SagorSarker)", "sagorsarker/bangla-bert-base"),
    #("BanglaBERT", "csebuetnlp/banglabert"),
    #("BanglaBERT Small", "csebuetnlp/banglabert_small"),
    #("BanglaBERT LARGE", "csebuetnlp/banglabert_large"),
    #("DistilBERT", "distilbert-base-multilingual-cased"),
    #("mBERT", "bert-base-multilingual-cased"),
    #("XLM-R Base", "xlm-roberta-base")
    ("Bangla_HateBERT", "saroarj/BanglaHateBert")
]

for n, m in models:
    train_and_eval(n, m)
