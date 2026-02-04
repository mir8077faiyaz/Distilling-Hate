import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score
)

from tabulate import tabulate
from thop import profile

# ✅ SAME NORMALIZER AS TRAINING
from normalizer import normalize


# ======================
# CONFIG  (FIXED)
# ======================
TEACHER_MODEL = "csebuetnlp/banglabert_large"   # FIX
STUDENT_MODEL = "csebuetnlp/banglabert_small"

MAX_LEN = 192
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# LOAD DATA
# ======================
def clean_bangla(text):
    text = normalize(str(text))                 # FIX
    text = re.sub(r"http\S+", "[URL]", text)
    return text.strip()


def load_tsv(path):
    df = pd.read_csv(path, sep="\t")

    # Drop rows where text is missing, but KEEP missing labels
    df = df.dropna(subset=["text"])

    # Normalize label text
    df["label"] = df["label"].astype(str).str.strip()

    # Convert missing / nan-like labels → Neutral
    df.loc[
        (df["label"].isna()) |
        (df["label"].str.lower().isin(["nan", "none"])) |
        (df["label"] == "") |
        (df["label"] == " "),
        "label"
    ] = "Neutral"

    df["text"] = df["text"].apply(clean_bangla)

    return df.reset_index(drop=True)

print("Loading dataset...")
train_df = load_tsv("train_1A.tsv")
dev_df   = load_tsv("dev_1A.tsv")
test_df  = load_tsv("test_1A.tsv")


# ======================
# LABEL ENCODING
# ======================
labels = sorted(train_df.label.unique())
label2id = {l: i for i, l in enumerate(labels)}
NUM_CLASSES = len(labels)

test_df["y"] = test_df.label.map(lambda x: label2id[str(x).strip()])


# ======================
# TOKENIZERS (FIXED)
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
        return self.texts[idx], int(self.labels[idx])


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
        "labels": torch.tensor(labels, dtype=torch.long)
    }


test_loader = DataLoader(
    HateDataset(test_df.text, test_df.y),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# ======================
# MODELS (FIXED)
# ======================

# ✅ EXACT STUDENT CLASS FROM TRAINING
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

# ✅ EXACT TEACHER MODEL FROM TRAINING
teacher = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_MODEL, num_labels=NUM_CLASSES
).to(DEVICE)
teacher.load_state_dict(torch.load("best_teacher_large.pt", map_location=DEVICE))

student = StudentClassifier().to(DEVICE)
student.load_state_dict(torch.load("best_student.pt", map_location=DEVICE))


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

            preds.extend(logits.argmax(dim=1).cpu().numpy())
            gold.extend(batch["labels"].numpy())

    return {
        "accuracy": accuracy_score(gold, preds),

        "macro_precision": precision_score(gold, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(gold, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(gold, preds, average="macro", zero_division=0),

        "micro_precision": precision_score(gold, preds, average="micro", zero_division=0),
        "micro_recall": recall_score(gold, preds, average="micro", zero_division=0),
        "micro_f1": f1_score(gold, preds, average="micro", zero_division=0),

        "weighted_precision": precision_score(gold, preds, average="weighted", zero_division=0),
        "weighted_recall": recall_score(gold, preds, average="weighted", zero_division=0),
        "weighted_f1": f1_score(gold, preds, average="weighted", zero_division=0),

        "per_class_precision": precision_score(gold, preds, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(gold, preds, average=None, zero_division=0).tolist(),
        "per_class_f1": f1_score(gold, preds, average=None, zero_division=0).tolist(),
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
# SUMMARY TABLE
# ======================
print("\n===== FULL METRICS =====\n")

summary = [
    ["Accuracy", teacher_results["accuracy"], student_results["accuracy"]],
    ["Macro Precision", teacher_results["macro_precision"], student_results["macro_precision"]],
    ["Macro Recall", teacher_results["macro_recall"], student_results["macro_recall"]],
    ["Macro F1", teacher_results["macro_f1"], student_results["macro_f1"]],
    ["Micro Precision", teacher_results["micro_precision"], student_results["micro_precision"]],
    ["Micro Recall", teacher_results["micro_recall"], student_results["micro_recall"]],
    ["Micro F1", teacher_results["micro_f1"], student_results["micro_f1"]],
    ["Weighted Precision", teacher_results["weighted_precision"], student_results["weighted_precision"]],
    ["Weighted Recall", teacher_results["weighted_recall"], student_results["weighted_recall"]],
    ["Weighted F1", teacher_results["weighted_f1"], student_results["weighted_f1"]],
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


# ======================
# PER CLASS
# ======================
print("\n===== PER CLASS (STUDENT) =====\n")
for i, label in enumerate(labels):
    print(f"{label:15s} | P={student_results['per_class_precision'][i]:.3f} "
          f"R={student_results['per_class_recall'][i]:.3f} "
          f"F1={student_results['per_class_f1'][i]:.3f}")


# ======================
# SAVE
# ======================
# json.dump(teacher_results, open("teacher_test_metrics.json", "w"), indent=2)
# json.dump(student_results, open("student_test_metrics.json", "w"), indent=2)

# pd.DataFrame({
#     "precision": student_results["per_class_precision"],
#     "recall": student_results["per_class_recall"],
#     "f1": student_results["per_class_f1"],
# }, index=labels).to_csv("student_per_class_test_metrics.csv")


# # ======================
# # PLOT
# # ======================
# plt.figure(figsize=(7,5))
# plt.bar(["Teacher", "Student"],
#         [teacher_results["macro_f1"], student_results["macro_f1"]])
# plt.ylabel("Macro-F1")
# plt.title("Teacher vs Student (TEST)")
# plt.savefig("teacher_vs_student_test.png")

print("\nDone.")
