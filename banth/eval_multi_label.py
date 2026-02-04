import os
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import hashlib
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

from tabulate import tabulate
from thop import profile
from datasets import load_dataset

from normalizer import normalize

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# CONFIG
# ======================
TEACHER_MODEL = "csebuetnlp/banglabert_large"
# TEACHER_MODEL = "bert-base-multilingual-cased"
STUDENT_MODEL = "csebuetnlp/banglabert_small"

MAX_LEN = 192
BATCH_SIZE = 32

LABEL_COLS = [
    "Political", "Religious", "Gender", "Personal Offense",
    "Abusive/Violence", "Origin", "Body Shaming", "Misc"
]
NUM_CLASSES = len(LABEL_COLS)

# ======================
# LOAD DATA (HF)
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

test_df = process_split(ds["test"])

# ======================
# TOKENIZERS
# ======================
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)

# ======================
# DATASET
# ======================
class HateDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], torch.tensor(self.labels[idx], dtype=torch.float)


def collate_fn(batch):
    texts, labels = zip(*batch)

    stu = student_tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    tea = teacher_tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    return {
        "student_input_ids": stu["input_ids"],
        "student_attention_mask": stu["attention_mask"],
        "teacher_input_ids": tea["input_ids"],
        "teacher_attention_mask": tea["attention_mask"],
        "labels": torch.stack(labels)
    }


test_loader = DataLoader(
    HateDataset(test_df.text, test_df.y),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# ======================
# MODELS
# ======================
class StudentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(STUDENT_MODEL)
        self.fc = nn.Linear(self.encoder.config.hidden_size, NUM_CLASSES)

    def forward(self, ids, mask):
        out = self.encoder(ids, mask)
        cls = out.last_hidden_state[:, 0]
        return self.fc(cls)


print("\nLoading models...")

teacher = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_MODEL,
    num_labels=NUM_CLASSES,
    problem_type="multi_label_classification"
).to(DEVICE)
teacher.load_state_dict(torch.load("best_teacher_large.pt", map_location=DEVICE))

student = StudentClassifier().to(DEVICE)
student.load_state_dict(torch.load("best_student.pt", map_location=DEVICE))
print(hashlib.md5(open("best_student.pt","rb").read()).hexdigest())
# ======================
# EVALUATION
# ======================
def evaluate(model, loader, teacher_mode=False):
    model.eval()
    preds, gold = [], []

    with torch.no_grad():
        for batch in loader:
            if teacher_mode:
                logits = model(
                    batch["teacher_input_ids"].to(DEVICE),
                    batch["teacher_attention_mask"].to(DEVICE)
                ).logits
            else:
                logits = model(
                    batch["student_input_ids"].to(DEVICE),
                    batch["student_attention_mask"].to(DEVICE)
                )

            p = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            preds.append(p)
            gold.append(batch["labels"].numpy())

    preds = np.vstack(preds)
    gold = np.vstack(gold)

    return {
        "f1_multi": f1_score(gold, preds, average="macro"),
        "subset_acc": accuracy_score(gold, preds),
        "hamming": hamming_loss(gold, preds),
    }

# ======================
# PARAMS & FLOPs
# ======================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model, tokenizer):
    model.eval()
    ids = torch.randint(0, tokenizer.vocab_size, (1, MAX_LEN)).to(DEVICE)
    mask = torch.ones_like(ids).to(DEVICE)
    flops, _ = profile(model, inputs=(ids, mask), verbose=False)
    return flops

# ======================
# RUN
# ======================
print("\nEvaluating TEACHER...")
teacher_results = evaluate(teacher, test_loader, teacher_mode=True)

print("\nEvaluating STUDENT...")
student_results = evaluate(student, test_loader)

print("\nComputing FLOPs...")
teacher_flops = compute_flops(teacher, teacher_tokenizer)
student_flops = compute_flops(student, student_tokenizer)

# ======================
# SUMMARY TABLE (PAPER METRICS)
# ======================
print("\n===== MULTI-LABEL METRICS =====\n")

summary = [
    ["F1 (Multi-Label)", teacher_results["f1_multi"], student_results["f1_multi"]],
    ["Subset Accuracy", teacher_results["subset_acc"], student_results["subset_acc"]],
    ["Hamming Loss", teacher_results["hamming"], student_results["hamming"]],
]

print(tabulate(summary, headers=["Metric", "Teacher", "Student"], floatfmt=".4f"))

# ======================
# MODEL INFO
# ======================
info = [
    ["Params", count_parameters(teacher), count_parameters(student)],
    ["FLOPs (1×)", f"{teacher_flops/1e9:.2f} GF", f"{student_flops/1e9:.2f} GF"],
    ["Architecture", TEACHER_MODEL, STUDENT_MODEL],
]

print("\n===== MODEL INFO =====\n")
print(tabulate(info, headers=["Property", "Teacher", "Student"]))

print("\nDone.")
