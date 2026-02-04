# class_counts.py
import re
import pandas as pd

# SAME NORMALIZER AS TRAINING
from normalizer import normalize


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


def print_class_counts(name, df):
    print(f"\n{name} class counts:")
    counts = df["label"].value_counts().sort_index()
    for label, cnt in counts.items():
        print(f"{label:15s} {cnt}")
    print(f"Total: {counts.sum()}")


train_df = load_tsv("train_1B.tsv")
dev_df   = load_tsv("dev_1B.tsv")
test_df  = load_tsv("test_1B.tsv")

print_class_counts("TRAIN", train_df)
print_class_counts("DEV", dev_df)
print_class_counts("TEST", test_df)
