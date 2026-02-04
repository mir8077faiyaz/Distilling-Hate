import os
import re
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
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
TEACHER_EPOCHS = 6
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# DATA
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
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ======================
# STAGE 1 — TEACHER TRAINING
# ======================
# ======================
# TEACHER OPTUNA OBJECTIVE
# ======================
def teacher_objective(trial):
    # Suggest Gamma: 0.0 effectively makes it Cross-Entropy
    gamma = trial.suggest_float("teacher_gamma", 0.0, 3.0)
    lr = trial.suggest_float("teacher_lr", 1e-5, 5e-5, log=True)
    
    print(f"\n--- Teacher Trial: Gamma={gamma:.2f}, LR={lr:.2e} ---")

    model = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL, num_labels=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = FocalLoss(gamma=gamma)

    best_trial_macro = 0

    for epoch in range(TEACHER_EPOCHS):
        model.train()
        for batch in train_loader:
            ids, mask, y = batch["t_ids"].to(DEVICE), batch["t_mask"].to(DEVICE), batch["labels"].to(DEVICE)
            logits = model(ids, mask).logits
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for batch in dev_loader:
                logits = model(batch["t_ids"].to(DEVICE), batch["t_mask"].to(DEVICE)).logits
                preds.extend(logits.argmax(1).cpu().numpy())
                gold.extend(batch["labels"].numpy())

        macro = f1_score(gold, preds, average="macro")
        
        # Check if this is the best version within THIS trial
        if macro > best_trial_macro:
            best_trial_macro = macro
            # Temporary save for the best within this trial
            torch.save(model.state_dict(), f"temp_teacher_trial_{trial.number}.pt")

    return best_trial_macro


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

    ce = FocalLoss(params["focal_gamma"]) if params["use_focal"] else nn.CrossEntropyLoss()
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


# ======================
# OPTUNA
# ======================
def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 2e-5, 6e-5, log=True),
        "temperature": trial.suggest_int("temperature", 4, 16),
        "lambda_kd": trial.suggest_float("lambda_kd", 0.3, 0.8),
        "lambda_hidden": trial.suggest_float("lambda_hidden", 0.1, 0.5),
        # "epochs": trial.suggest_int("epochs", 6, 10),
        "epochs": trial.suggest_int("epochs", 4, 6),
        "warmup": trial.suggest_float("warmup", 0.05, 0.15),
        "use_focal": trial.suggest_categorical("use_focal", [True, False]),
        "focal_gamma": trial.suggest_int("focal_gamma", 1, 3),
    }
    return run_kd(params)


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    # --- STAGE 1: OPTIMIZE TEACHER ---
    print("=== OPTIMIZING TEACHER (FOCAL VS CROSS-ENTROPY) ===")
    teacher_study = optuna.create_study(direction="maximize")
    # 5-10 trials are usually enough to determine if Focal helps over CE
    teacher_study.optimize(teacher_objective, n_trials=8) 
    
    print("\nBest Teacher Params:", teacher_study.best_params)
    
    # Identify the best trial and rename its saved model to the permanent file
    best_t_trial = teacher_study.best_trial.number
    best_teacher_path = "best_teacher_large.pt"
    os.rename(f"temp_teacher_trial_{best_t_trial}.pt", best_teacher_path)
    print(f"Saved best teacher model from Trial {best_t_trial}")

    # Cleanup other temporary teacher files
    for f in os.listdir():
        if f.startswith("temp_teacher_trial_") and f.endswith(".pt"):
            os.remove(f)

    # --- STAGE 2: OPTIMIZE STUDENT (KD) ---
    print("\n=== STARTING STUDENT KNOWLEDGE DISTILLATION ===")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    print("\nBest Student Params:", study.best_trial.params)

    # --- FINAL EVALUATION ---
    # Load the best student found during the 15 trials
    student = StudentClassifier().to(DEVICE)
    student.load_state_dict(torch.load("best_student.pt"))

    acc, macro, micro = evaluate(student, test_loader)
    print("\n=== FINAL STUDENT TEST PERFORMANCE ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro:.4f}")
    print(f"Micro F1: {micro:.4f}")
