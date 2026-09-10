"""
PhishShield AI — OOD False-Positive Evaluation

Measures how often the Layer 2 classifier calls a LEGITIMATE email phishing,
on emails drawn from a distribution the model was never trained on.

Why this exists:
    Internal test accuracy (~99%) is measured on held-out data from the SAME
    sources as training. It says nothing about emails from a different world:
    university circulars, HR notices, non-US professional English. A single
    anecdotal false positive isn't measurable progress — this turns it into a
    rate you can improve against and report honestly.

This script does NOT train, tune, or modify anything. It only measures.
Run it BEFORE adding data (baseline) and AFTER retraining (comparison).

Input CSV needs a `text` column. Every row is assumed LEGITIMATE (label 0)
unless a `label` column says otherwise. An optional `category` column lets
you break the false-positive rate down by kind of email, which is how you
find out WHICH register the model fails on.

Usage:
    # Baseline on your current deployed model
    python evaluate_ood.py --model Layer-2/models/phishing-model-3class \
                           --csv ood_legit.csv --out results/ood_baseline

    # After retraining, same command with the new model + --label to compare
    python evaluate_ood.py --model Layer-2/models/phishing-model-v3 \
                           --csv ood_legit.csv --out results/ood_v3

Outputs: a printed report + ood_report.json + misclassified.csv (every
false positive with its confidence, so you can inspect WHY).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_label_map(model_dir: Path) -> dict:
    p = model_dir / "label_map.json"
    if p.exists():
        return {int(k): v for k, v in json.loads(p.read_text()).items()}
    return {0: "ham", 1: "phishing", 2: "ai_phish"}


def score_texts(model_dir: Path, texts: list[str], max_length: int = 256,
                batch: int = 16):
    """Run every text through the classifier; return predictions + probabilities."""
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device).eval()

    preds, probs = [], []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()
        preds.extend(p.argmax(axis=1).tolist())
        probs.extend(p.tolist())
        print(f"  scored {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.array(preds), np.array(probs)


def main():
    ap = argparse.ArgumentParser(description="OOD false-positive evaluation")
    ap.add_argument("--model", required=True, help="Path to the model folder")
    ap.add_argument("--csv", required=True, help="CSV of legitimate emails (needs `text`)")
    ap.add_argument("--out", default="results/ood", help="Output directory")
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    model_dir = Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label_names = load_label_map(model_dir)

    df = pd.read_csv(args.csv)
    if "text" not in df.columns:
        raise SystemExit(f"{args.csv} needs a `text` column. Got: {list(df.columns)}")
    df = df[df["text"].notna()].reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    if "label" not in df.columns:
        df["label"] = 0  # assume everything in this file is legitimate

    print(f"Scoring {len(df)} emails through {model_dir.name}...")
    preds, probs = score_texts(model_dir, df["text"].tolist(), args.max_length)

    df["predicted"] = [label_names.get(int(p), str(p)) for p in preds]
    df["p_ham"] = probs[:, 0]
    # "phishing-ish" = everything that isn't ham (covers binary and 3-class)
    df["p_not_ham"] = 1.0 - probs[:, 0]
    df["confidence"] = probs.max(axis=1)

    legit = df[df["label"] == 0]
    n = len(legit)
    fp = legit[legit["predicted"] != "ham"]
    fp_rate = len(fp) / n if n else 0.0

    print("\n" + "=" * 58)
    print("OOD FALSE-POSITIVE REPORT")
    print("=" * 58)
    print(f"Model:                 {model_dir.name}")
    print(f"Legitimate emails:     {n}")
    print(f"Called phishing:       {len(fp)}")
    print(f"FALSE POSITIVE RATE:   {fp_rate:.1%}")
    if len(fp):
        print(f"Mean confidence when wrong: {fp['confidence'].mean():.1%}")
        print(f"Max  confidence when wrong: {fp['confidence'].max():.1%}")
    print(f"Mean P(ham) across all:     {legit['p_ham'].mean():.1%}")

    # Per-category breakdown — this is how you find WHICH register fails
    by_cat = {}
    if "category" in df.columns:
        print("\nBy category:")
        for cat, g in legit.groupby("category"):
            g_fp = g[g["predicted"] != "ham"]
            rate = len(g_fp) / len(g) if len(g) else 0.0
            by_cat[str(cat)] = {
                "n": int(len(g)), "false_positives": int(len(g_fp)),
                "fp_rate": float(rate), "mean_p_ham": float(g["p_ham"].mean()),
            }
            print(f"  {str(cat):32s} {len(g_fp):3d}/{len(g):3d}  ({rate:5.1%})")

    # Predicted-class spread
    print("\nPredicted class distribution:")
    for cls, cnt in legit["predicted"].value_counts().items():
        print(f"  {cls:12s} {cnt:4d}  ({cnt/n:5.1%})")

    report = {
        "model": str(model_dir),
        "n_legitimate": int(n),
        "false_positives": int(len(fp)),
        "false_positive_rate": float(fp_rate),
        "mean_confidence_when_wrong": float(fp["confidence"].mean()) if len(fp) else None,
        "mean_p_ham": float(legit["p_ham"].mean()),
        "by_category": by_cat,
        "predicted_distribution": {str(k): int(v) for k, v in legit["predicted"].value_counts().items()},
    }
    (out / "ood_report.json").write_text(json.dumps(report, indent=2))

    if len(fp):
        cols = [c for c in ["category", "predicted", "confidence", "p_ham", "text"] if c in fp.columns]
        fp_out = fp[cols].copy()
        fp_out["text"] = fp_out["text"].str.slice(0, 400)
        fp_out.sort_values("confidence", ascending=False).to_csv(
            out / "misclassified.csv", index=False)
        print(f"\nWrote {len(fp)} misclassified emails to {out / 'misclassified.csv'}")
        print("Inspect that file to see WHAT the model is reacting to.")

    print(f"Wrote report to {out / 'ood_report.json'}")
    print("\nThis number is your baseline. Re-run after retraining to show improvement.")


if __name__ == "__main__":
    main()