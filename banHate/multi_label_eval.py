import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from sklearn.metrics import (
    precision_score, recall_score, f1_score, hamming_loss
)

from tabulate import tabulate
from thop import profile

from normalizer import normalize

# ======================
# CONFIG
# ======================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_MODEL = "csebuetnlp/banglabert_large"
STUDENT_MODEL = "csebuetnlp/banglabert_small"

MAX_LEN = 192
BATCH_SIZE = 32
THRESHOLD = 0.1


# ======================
# DATA (HUGGINGFACE – FIXED TO SCHEMA)
# ======================
def clean_text(text):
    text = normalize(str(text))
    text = re.sub(r"http\S+", "[URL]", text)
    return text.strip()


ds = load_dataset("aplycaebous/BanTH")

LABEL_COLS = [
    "Political",
    "Religious",
    "Gender",
    "Personal Offense",
    "Abusive/Violence",
    "Origin",
    "Body Shaming",
    "Misc",
]

NUM_CLASSES = len(LABEL_COLS)


def prepare_split(split):
    texts = [clean_text(x) for x in split["Text"]]
    labels = np.stack([split[c] for c in LABEL_COLS], axis=1).astype("float32")
    return texts, labels


test_texts, test_labels = prepare_split(ds["test"])


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
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], torch.tensor(self.labels[idx])


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
    HateDataset(test_texts, test_labels),
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
    TEACHER_MODEL, num_labels=NUM_CLASSES
).to(DEVICE)
teacher.load_state_dict(torch.load("best_teacher_large.pt", map_location=DEVICE))

student = StudentClassifier().to(DEVICE)
student.load_state_dict(torch.load("best_student.pt", map_location=DEVICE))


# ======================
# EVALUATION (MULTI-LABEL + HAMMING)
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

            probs = torch.sigmoid(logits)
            preds.append((probs > THRESHOLD).cpu().numpy())
            gold.append(batch["labels"].cpu().numpy())

    preds = np.vstack(preds)
    gold = np.vstack(gold)

    return {
        "macro_precision": precision_score(gold, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(gold, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(gold, preds, average="macro", zero_division=0),

        "micro_precision": precision_score(gold, preds, average="micro", zero_division=0),
        "micro_recall": recall_score(gold, preds, average="micro", zero_division=0),
        "micro_f1": f1_score(gold, preds, average="micro", zero_division=0),

        "samples_f1": f1_score(gold, preds, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(gold, preds),

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
    ["Macro Precision", teacher_results["macro_precision"], student_results["macro_precision"]],
    ["Macro Recall", teacher_results["macro_recall"], student_results["macro_recall"]],
    ["Macro F1", teacher_results["macro_f1"], student_results["macro_f1"]],
    ["Micro Precision", teacher_results["micro_precision"], student_results["micro_precision"]],
    ["Micro Recall", teacher_results["micro_recall"], student_results["micro_recall"]],
    ["Micro F1", teacher_results["micro_f1"], student_results["micro_f1"]],
    ["Samples F1", teacher_results["samples_f1"], student_results["samples_f1"]],
    ["Hamming Loss ↓", teacher_results["hamming_loss"], student_results["hamming_loss"]],
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
# PER LABEL (STUDENT)
# ======================
print("\n===== PER LABEL (STUDENT) =====\n")
for i, label in enumerate(LABEL_COLS):
    print(f"{label:18s} | "
          f"P={student_results['per_class_precision'][i]:.3f} "
          f"R={student_results['per_class_recall'][i]:.3f} "
          f"F1={student_results['per_class_f1'][i]:.3f}")

print("\nDone.")
