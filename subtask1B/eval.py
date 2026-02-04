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
train_df = load_tsv("train_1B.tsv")
dev_df   = load_tsv("dev_1B.tsv")
test_df  = load_tsv("test_1B.tsv")


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


from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # --- Subtle, single-hue palette ---
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_title("Student Model subtask-1B", fontsize=18, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # Light white grid, non-aggressive
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    thresh = cm_norm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{cm[i,j]}\n{cm_norm[i,j]*100:.1f}%"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=9,
                color="white" if cm_norm[i,j] > thresh else "black",
                fontweight="semibold"
            )

    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
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
