import os
import re
import json
import numpy as np
import pandas as pd

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

from sklearn.metrics import f1_score, accuracy_score
import optuna
from normalizer import normalize


# ======================
# CONFIG
# ======================
TEACHER_MODEL = "csebuetnlp/banglabert_large"
STUDENT_MODEL = "csebuetnlp/banglabert_small"

MAX_LEN = 192
BATCH_SIZE = 32
TEACHER_EPOCHS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# DATA
# ======================
def clean_bangla(text):
    text = normalize(str(text))
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



train_df = load_tsv("train_1A.tsv")
dev_df   = load_tsv("dev_1A.tsv")
test_df  = load_tsv("test_1A.tsv")

labels = sorted(train_df.label.unique())
label2id = {l: i for i, l in enumerate(labels)}
NUM_CLASSES = len(labels)
print("Labels:", labels)
print()

for l in labels:
    count = (train_df["label"] == l).sum()
    print(f"{l}: {count}")

for df in [train_df, dev_df, test_df]:
    df["y"] = df.label.map(lambda x: label2id[str(x).strip()])


# ======================
# DATASET
# ======================
tokenizer_teacher = AutoTokenizer.from_pretrained(TEACHER_MODEL)
tokenizer_student = AutoTokenizer.from_pretrained(STUDENT_MODEL)


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

    t = tokenizer_teacher(
        list(texts), padding="max_length", truncation=True,
        max_length=MAX_LEN, return_tensors="pt"
    )

    s = tokenizer_student(
        list(texts), padding="max_length", truncation=True,
        max_length=MAX_LEN, return_tensors="pt"
    )

    return {
        "t_ids": t["input_ids"],
        "t_mask": t["attention_mask"],
        "s_ids": s["input_ids"],
        "s_mask": s["attention_mask"],
        "labels": torch.tensor(labels)
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
# LOSSES
# ======================
class WeightedFocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.weights = weights   # torch tensor on DEVICE

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=self.weights
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

class_names = labels  # already sorted
weights = compute_class_weight(
    'balanced',
    classes=np.array(class_names),
    y=train_df['label']
)

# map correctly
ordered = [weights[list(class_names).index(l)] for l in labels]
weights = torch.tensor(ordered, dtype=torch.float).to(DEVICE)


# ======================
# STAGE 1 — TEACHER TRAINING
# ======================
def train_teacher():
    print("\n=== TRAINING LARGE TEACHER (FOCAL) ===")

    model = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL, num_labels=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = WeightedFocalLoss(weights=weights, gamma=2)

    best_score = 0

    for epoch in range(TEACHER_EPOCHS):
        model.train()
        for batch in train_loader:
            ids = batch["t_ids"].to(DEVICE)
            mask = batch["t_mask"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            logits = model(ids, mask).logits
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for batch in dev_loader:
                logits = model(
                    batch["t_ids"].to(DEVICE),
                    batch["t_mask"].to(DEVICE)
                ).logits
                preds.extend(logits.argmax(1).cpu().numpy())
                gold.extend(batch["labels"].numpy())

        macro = f1_score(gold, preds, average="macro")
        micro = f1_score(gold, preds, average="micro")
        score = macro + micro

        print(f"Epoch {epoch+1} | Macro={macro:.4f} Micro={micro:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_teacher_large.pt")

    print("Teacher training done.")


# ======================
# STUDENT MODEL
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


# ======================
# KD TRAINING
# ======================
def run_kd(params):
    teacher = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL, num_labels=NUM_CLASSES
    ).to(DEVICE)
    teacher.load_state_dict(torch.load("best_teacher_large.pt"))
    teacher.eval()

    student = StudentClassifier().to(DEVICE)

    proj = nn.Linear(1024, student.encoder.config.hidden_size).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(proj.parameters()),
        lr=params["lr"]
    )

    total_steps = len(train_loader) * params["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * params["warmup"]), total_steps
    )

    ce = WeightedFocalLoss(weights=weights, gamma=params["focal_gamma"]) \
     if params["use_focal"] else nn.CrossEntropyLoss()

    kl = nn.KLDivLoss(reduction="batchmean")

    best = 0

    for epoch in range(params["epochs"]):
        student.train()

        for batch in train_loader:
            s_ids = batch["s_ids"].to(DEVICE)
            s_mask = batch["s_mask"].to(DEVICE)
            t_ids = batch["t_ids"].to(DEVICE)
            t_mask = batch["t_mask"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            with torch.no_grad():
                t_out = teacher(t_ids, t_mask, output_hidden_states=True)
                t_logits = t_out.logits
                t_hidden = proj(t_out.hidden_states[-1][:, 0])

            s_logits, s_hidden = student(s_ids, s_mask, return_hidden=True)

            loss_ce = ce(s_logits, y)
            loss_kd = kl(
                F.log_softmax(s_logits / params["temperature"], 1),
                F.softmax(t_logits / params["temperature"], 1)
            ) * params["temperature"] ** 2

            loss_h = F.mse_loss(s_hidden, t_hidden)

            loss = (
                (1 - params["lambda_kd"]) * loss_ce +
                params["lambda_kd"] * loss_kd +
                params["lambda_hidden"] * loss_h
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        acc, macro, micro = evaluate(student, dev_loader)
        score = macro + micro
        if score > best:
            best = score
            torch.save(student.state_dict(), "best_student.pt")

    return best


# ======================
# EVALUATE
# ======================
def evaluate(model, loader):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["s_ids"].to(DEVICE),
                batch["s_mask"].to(DEVICE)
            )
            preds.extend(logits.argmax(1).cpu().numpy())
            gold.extend(batch["labels"].numpy())

    return (
        accuracy_score(gold, preds),
        f1_score(gold, preds, average="macro"),
        f1_score(gold, preds, average="micro")
    )


def collect_probs(model, loader):
    model.eval()
    probs, gold = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["s_ids"].to(DEVICE),
                batch["s_mask"].to(DEVICE)
            )
            p = F.softmax(logits, dim=-1).cpu().numpy()
            probs.append(p)
            gold.extend(batch["labels"].numpy())
    return np.vstack(probs), np.array(gold)

def apply_thresholds(probs, thresholds):
    preds = []

    for p in probs:
        assigned = False
        order = np.argsort(p)[::-1]   # high → low

        for cls in order:
            if p[cls] >= thresholds[cls]:
                preds.append(cls)
                assigned = True
                break

        if not assigned:
            preds.append(np.argmax(p))   # fallback

    return np.array(preds)

def threshold_objective(trial, dev_probs, dev_gold):
    thresholds = []
    for c in range(NUM_CLASSES):
        thresholds.append(trial.suggest_float(f"t{c}", 0.1, 0.9))

    preds = apply_thresholds(dev_probs, thresholds)
    return f1_score(dev_gold, preds, average="macro")


# ======================
# OPTUNA
# ======================
def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 2e-5, 6e-5, log=True),
        "temperature": trial.suggest_int("temperature", 4, 16),
        "lambda_kd": trial.suggest_float("lambda_kd", 0.3, 0.8),
        "lambda_hidden": trial.suggest_float("lambda_hidden", 0.1, 0.5),
        "epochs": trial.suggest_int("epochs", 6, 10),
        "warmup": trial.suggest_float("warmup", 0.05, 0.15),
        "use_focal": trial.suggest_categorical("use_focal", [True, False]),
        "focal_gamma": trial.suggest_int("focal_gamma", 1, 3),
    }
    return run_kd(params)


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    train_teacher()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    print("Best params:", study.best_trial.params)

    # student = StudentClassifier().to(DEVICE)
    # student.load_state_dict(torch.load("best_student.pt"))

    # acc, macro, micro = evaluate(student, test_loader)
    # print("\n=== FINAL STUDENT TEST ===")
    # print("Accuracy:", acc)
    # print("Macro F1:", macro)
    # print("Micro F1:", micro)
    student = StudentClassifier().to(DEVICE)
    student.load_state_dict(torch.load("best_student.pt"))

    print("\nCollecting Dev probabilities...")
    dev_probs, dev_gold = collect_probs(student, dev_loader)

    print("\n=== OPTIMIZING THRESHOLDS WITH OPTUNA ===")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: threshold_objective(t, dev_probs, dev_gold), n_trials=200)

    best_thresholds = [study.best_params[f"t{i}"] for i in range(NUM_CLASSES)]
    print("\nBest thresholds:", best_thresholds)

    print("\nEvaluating on TEST with thresholds...")

    test_probs, test_gold = collect_probs(student, test_loader)
    test_preds = apply_thresholds(test_probs, best_thresholds)

    acc = accuracy_score(test_gold, test_preds)
    macro = f1_score(test_gold, test_preds, average="macro")
    micro = f1_score(test_gold, test_preds, average="micro")

    print("\n=== FINAL STUDENT TEST (THRESHOLD TUNED) ===")
    print("Accuracy:", acc)
    print("Macro F1:", macro)
    print("Micro F1:", micro)

    from sklearn.metrics import classification_report
    print("\n=== PER CLASS ===")
    print(classification_report(test_gold, test_preds, target_names=labels, digits=4))

