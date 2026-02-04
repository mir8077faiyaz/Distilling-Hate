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
    accuracy_score, confusion_matrix
)

from tabulate import tabulate
from thop import profile
from datasets import load_dataset

# ✅ SAME NORMALIZER AS TRAINING
from normalizer import normalize

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# CONFIG (FIXED)
# ======================
TEACHER_MODEL = "csebuetnlp/banglabert_large"
STUDENT_MODEL = "csebuetnlp/banglabert_small"

MAX_LEN = 192
BATCH_SIZE = 32

# ======================
# LOAD DATA (HF + BINARY)
# ======================
def clean_bangla(text):
    text = normalize(str(text))
    text = re.sub(r"http\S+", "[URL]", text)
    return text.strip()

print("Loading HuggingFace dataset...")
ds = load_dataset("aplycaebous/BanTH")

def process_split(split):
    df = split.to_pandas()
    df = df.dropna(subset=["Text", "Label"])
    df["text"] = df["Text"].astype(str).apply(clean_bangla)
    df["y"] = df["Label"].astype(int)
    return df.reset_index(drop=True)

train_df = process_split(ds["train"])
dev_df = process_split(ds["validation"])
test_df = process_split(ds["test"])

labels = ["0", "1"]
NUM_CLASSES = 2

print("Labels:", labels)
print(train_df["y"].value_counts())

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
# CONFUSION MATRIX
# ======================
def save_student_confusion_matrix(model, loader, labels, filename="student_confusion_matrix.pdf"):
    model.eval()
    preds, gold = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["student_input_ids"].to(DEVICE),
                batch["student_attention_mask"].to(DEVICE)
            )
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            gold.extend(batch["labels"].numpy())

    cm = confusion_matrix(gold, preds, labels=range(len(labels)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_title("Student Model (Binary)", fontsize=16)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i,j]}\n{cm_norm[i,j]*100:.1f}%",
                    ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(filename, format="pdf")
    plt.close()


save_student_confusion_matrix(student, test_loader, labels)

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
    print(f"{label:5s} | P={student_results['per_class_precision'][i]:.3f} "
          f"R={student_results['per_class_recall'][i]:.3f} "
          f"F1={student_results['per_class_f1'][i]:.3f}")

print("\nDone.")
