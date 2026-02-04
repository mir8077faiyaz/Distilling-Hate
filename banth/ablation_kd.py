import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score
)

from tabulate import tabulate
from thop import profile
from datasets import load_dataset

from normalizer import normalize


# ======================
# CONFIG
# ======================
TEACHER_MODEL = "csebuetnlp/banglabert_large"
STUDENT_MODEL = "csebuetnlp/banglabert_small"

MAX_LEN = 192
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# DATA (HF + BINARY)
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

student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)


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
        "student_ids": stu["input_ids"],
        "student_mask": stu["attention_mask"],
        "teacher_ids": tea["input_ids"],
        "teacher_mask": tea["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long)
    }


train_loader = DataLoader(
    HateDataset(train_df.text, train_df.y),
    batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    HateDataset(dev_df.text, dev_df.y),
    batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    HateDataset(test_df.text, test_df.y),
    batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# ======================
# STUDENT + LOSSES
# ======================
class StudentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(STUDENT_MODEL)
        self.fc = nn.Linear(self.encoder.config.hidden_size, NUM_CLASSES)

    def forward(self, ids, mask, return_hidden=False):
        out = self.encoder(ids, mask, output_hidden_states=True)
        cls = out.last_hidden_state[:, 0]
        logits = self.fc(cls)
        if return_hidden:
            return logits, cls
        return logits


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ======================
# FULL METRICS EVAL
# ======================
def evaluate_full(student_model):
    teacher = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL, num_labels=NUM_CLASSES
    ).to(DEVICE)
    teacher.load_state_dict(torch.load("best_teacher_large.pt", map_location=DEVICE))

    teacher.eval()
    student_model.eval()

    preds_s, preds_t, gold = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            gold.extend(batch["labels"].numpy())

            t_logits = teacher(
                batch["teacher_ids"].to(DEVICE),
                batch["teacher_mask"].to(DEVICE)
            ).logits

            s_logits = student_model(
                batch["student_ids"].to(DEVICE),
                batch["student_mask"].to(DEVICE)
            )

            preds_t.extend(t_logits.argmax(1).cpu().numpy())
            preds_s.extend(s_logits.argmax(1).cpu().numpy())

    def m(p):
        return {
            "acc": accuracy_score(gold, p),
            "macro_f1": f1_score(gold, p, average="macro"),
            "micro_f1": f1_score(gold, p, average="micro"),
            "weighted_f1": f1_score(gold, p, average="weighted"),
        }

    T = m(preds_t)
    S = m(preds_s)

    teacher_flops, _ = profile(
        teacher,
        inputs=(torch.randint(0, 1000, (1, MAX_LEN)).to(DEVICE),
                torch.ones(1, MAX_LEN).to(DEVICE)),
        verbose=False
    )

    student_flops, _ = profile(
        student_model,
        inputs=(torch.randint(0, 1000, (1, MAX_LEN)).to(DEVICE),
                torch.ones(1, MAX_LEN).to(DEVICE)),
        verbose=False
    )

    print("\n===== FULL METRICS =====\n")
    table = [
        ["Accuracy", T["acc"], S["acc"]],
        ["Macro F1", T["macro_f1"], S["macro_f1"]],
        ["Micro F1", T["micro_f1"], S["micro_f1"]],
        ["Weighted F1", T["weighted_f1"], S["weighted_f1"]],
    ]
    print(tabulate(table, headers=["Metric", "Teacher", "Student"], floatfmt=".4f"))

    print("\nTeacher FLOPS:", teacher_flops / 1e9, "GF")
    print("Student FLOPS:", student_flops / 1e9, "GF")


# ======================
# KD TRAIN LOOP
# ======================
def run_single(name, lambda_kd, use_focal, lambda_hidden):

    temperature = 8
    epochs = 6
    print(f"\n===== TRAINING {name} =====")

    teacher = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL, num_labels=NUM_CLASSES
    ).to(DEVICE)
    teacher.load_state_dict(torch.load("best_teacher_large.pt"))
    teacher.eval()

    student = StudentClassifier().to(DEVICE)
    proj = nn.Linear(1024, student.encoder.config.hidden_size).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(proj.parameters()),
        lr=2e-5
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps
    )

    ce = FocalLoss(2) if use_focal else nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    best = -1
    best_state = None

    for epoch in range(epochs):
        student.train()

        for batch in train_loader:
            s_ids = batch["student_ids"].to(DEVICE)
            s_mask = batch["student_mask"].to(DEVICE)
            t_ids = batch["teacher_ids"].to(DEVICE)
            t_mask = batch["teacher_mask"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            with torch.no_grad():
                t_out = teacher(t_ids, t_mask, output_hidden_states=True)
                t_logits = t_out.logits
                t_hidden = proj(t_out.hidden_states[-1][:, 0])

            s_logits, s_hidden = student(s_ids, s_mask, return_hidden=True)

            loss_ce = ce(s_logits, y)
            loss_kd = kl(
                F.log_softmax(s_logits / temperature, 1),
                F.softmax(t_logits / temperature, 1)
            ) * (temperature ** 2)

            loss_h = F.mse_loss(s_hidden, t_hidden)

            loss = (
                (1 - lambda_kd) * loss_ce +
                lambda_kd * loss_kd +
                lambda_hidden * loss_h
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        student.eval()
        preds, gold = [], []
        with torch.no_grad():
            for batch in dev_loader:
                logits = student(
                    batch["student_ids"].to(DEVICE),
                    batch["student_mask"].to(DEVICE)
                )
                preds.extend(logits.argmax(1).cpu().numpy())
                gold.extend(batch["labels"].numpy())

        macro_f = f1_score(gold, preds, average="macro")
        if macro_f > best:
            best = macro_f
            best_state = student.state_dict()

    student.load_state_dict(best_state)
    return student


# ======================
# RUN ABLATIONS
# ======================
def run_suite():
    configs = [
        ("A1_lambda0_focal_hidden", 0, True, 0.3),
        ("A2_lambda0_ce_hidden", 0, False, 0.3),
        ("A3_lambda0_focal_only", 0, True, 0.0),
        ("A4_lambda0_ce_only", 0, False, 0.0),
        ("A5_lambda1_kd_mse", 1, False, 0.3),
        ("A6_lambda1_kd_only", 1, False, 0.0),
    ]

    for name, lk, focal, lh in configs:
        student_model = run_single(name, lk, focal, lh)
        evaluate_full(student_model)


if __name__ == "__main__":
    assert os.path.exists("best_teacher_large.pt"), \
        "Teacher checkpoint best_teacher_large.pt required"
    run_suite()
