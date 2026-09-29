import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL = Path(r".\Layer-2\models\phishing-model-3class")
DATA = Path(r".\Layer-2\data\dataset_final\val.csv")
OUT = Path(r".\results\v4_validation_probs.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

df = pd.read_csv(DATA)
df = df[["text", "label"]].dropna().reset_index(drop=True)
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL))
model.to(device)
model.eval()

all_probs = []

batch_size = 16
max_length = 256

for i in range(0, len(df), batch_size):
    texts = df["text"].iloc[i:i + batch_size].tolist()

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    all_probs.extend(probs.tolist())

    print(f"Scored {min(i + batch_size, len(df))}/{len(df)}", end="\r")

print()

probs = np.array(all_probs)

df["p_ham"] = probs[:, 0]
df["p_phishing"] = probs[:, 1]
df["p_ai_phish"] = probs[:, 2]
df["predicted"] = probs.argmax(axis=1)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f"Saved: {OUT}")
print(f"Rows: {len(df)}")