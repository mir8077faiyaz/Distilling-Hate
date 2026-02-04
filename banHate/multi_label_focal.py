import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import f1_score, hamming_loss
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

THRESHOLD = 0.5
# ======================
# DATA (HUGGINGFACE)
# ======================
def clean_text(text):
    text = normalize(str(text))
    text = re.sub(r"http\S+", "[URL]", text)
    return text.strip()


ds = load_dataset("aplycaebous/BanTH")

# UPDATED: aligned with HF schema (including Misc, Abusive/Violence naming)
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


train_texts, train_labels = prepare_split(ds["train"])
dev_texts, dev_labels     = prepare_split(ds["validation"])
test_texts, test_labels   = prepare_split(ds["test"])


# ======================
# TOKENIZERS
# ======================
tokenizer_teacher = AutoTokenizer.from_pretrained(TEACHER_MODEL)
tokenizer_student = AutoTokenizer.from_pretrained(STUDENT_MODEL)


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
        "labels": torch.stack(labels)
    }


train_loader = DataLoader(
    HateDataset(train_texts, train_labels),
    batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    HateDataset(dev_texts, dev_labels),
    batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    HateDataset(test_texts, test_labels),
    batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# ======================
# MULTI-LABEL FOCAL LOSS
# ======================
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        return loss.mean()


# ======================
# TEACHER OPTUNA
# ======================
def teacher_objective(trial):
    gamma = trial.suggest_float("teacher_gamma", 0.0, 3.0)
    lr = trial.suggest_float("teacher_lr", 1e-5, 5e-5, log=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL, num_labels=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = MultiLabelFocalLoss(gamma)

    best = -1.0

    for _ in range(TEACHER_EPOCHS):
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

        metrics = evaluate_teacher(model, dev_loader)
        # if metrics["micro_f1"] > best:
        #     best = metrics["micro_f1"]
        #     torch.save(model.state_dict(), f"temp_teacher_{trial.number}.pt")
        score = metrics["micro_f1"] - metrics["hamming_loss"]
        if score > best:
            best = score
            torch.save(model.state_dict(), f"temp_teacher_{trial.number}.pt")

    return best


def evaluate_teacher(model, loader):
    model.eval()
    preds, gold = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["t_ids"].to(DEVICE),
                batch["t_mask"].to(DEVICE)
            ).logits

            probs = torch.sigmoid(logits)
            preds.append((probs > THRESHOLD).cpu().numpy())
            gold.append(batch["labels"].cpu().numpy())

    preds = np.vstack(preds)
    gold = np.vstack(gold)

    return {
        "macro_f1": f1_score(gold, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(gold, preds, average="micro", zero_division=0),
        "samples_f1": f1_score(gold, preds, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(gold, preds),
    }


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
        return (logits, cls) if return_hidden else logits


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

    sup_loss = MultiLabelFocalLoss(params["focal_gamma"])
    kl = nn.KLDivLoss(reduction="batchmean")

    best = -1.0

    for _ in range(params["epochs"]):
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

            loss_sup = sup_loss(s_logits, y)
            loss_kd = kl(
                torch.log(torch.sigmoid(s_logits / params["temperature"]) + 1e-8),
                torch.sigmoid(t_logits / params["temperature"])
            ) * params["temperature"] ** 2

            loss_h = F.mse_loss(s_hidden, t_hidden)

            loss = (
                (1 - params["lambda_kd"]) * loss_sup +
                params["lambda_kd"] * loss_kd +
                params["lambda_hidden"] * loss_h
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        metrics = evaluate_student(student, dev_loader)
        score = metrics["micro_f1"] - metrics["hamming_loss"]
        if score > best:
            best = score
            torch.save(student.state_dict(), "best_student.pt")
        # score = metrics["micro_f1"]
        # if score > best:
        #     best = score
        #     torch.save(student.state_dict(), "best_student.pt")

    return best


def evaluate_student(model, loader):
    model.eval()
    preds, gold = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["s_ids"].to(DEVICE),
                batch["s_mask"].to(DEVICE)
            )
            probs = torch.sigmoid(logits)
            preds.append((probs > THRESHOLD).cpu().numpy())
            gold.append(batch["labels"].cpu().numpy())

    preds = np.vstack(preds)
    gold = np.vstack(gold)

    return {
        "macro_f1": f1_score(gold, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(gold, preds, average="micro", zero_division=0),
        "samples_f1": f1_score(gold, preds, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(gold, preds),
    }


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    print("=== OPTIMIZING TEACHER ===")
    teacher_study = optuna.create_study(direction="maximize")
    teacher_study.optimize(teacher_objective, n_trials=6)

    best_trial = teacher_study.best_trial.number
    os.rename(f"temp_teacher_{best_trial}.pt", "best_teacher_large.pt")
    # print("=== USING PRETRAINED TEACHER ===")
    # assert os.path.exists("best_teacher_large.pt"), "Teacher checkpoint not found"

    print("=== KD TRAINING ===")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: run_kd({
            "lr": t.suggest_float("lr", 2e-5, 6e-5, log=True),
            "temperature": t.suggest_int("temperature", 4, 16),
            "lambda_kd": t.suggest_float("lambda_kd", 0.3, 0.8),
            "lambda_hidden": t.suggest_float("lambda_hidden", 0.1, 0.5),
            "epochs": t.suggest_int("epochs", 6, 8),
            "warmup": t.suggest_float("warmup", 0.05, 0.15),
            "focal_gamma": t.suggest_int("focal_gamma", 1, 3),
        }),
        n_trials=15
    )

    student = StudentClassifier().to(DEVICE)
    student.load_state_dict(torch.load("best_student.pt"))

    metrics = evaluate_student(student, test_loader)
    print("\n=== FINAL TEST PERFORMANCE ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
