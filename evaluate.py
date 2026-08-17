"""
PhishShield AI — Model Evaluation & Reporting

Generates the evaluation artifacts you'd put in a report or portfolio:
  1. Confusion matrix (heatmap PNG)
  2. Per-class precision / recall / F1 table (printed + saved JSON)
  3. Per-class metrics bar chart (PNG)
  4. Layer-comparison: Layer 1 rules alone vs Layer 2 model alone vs the
     fused pipeline — the chart that proves the hybrid design earns its keep.

Runs the trained Layer 2 model over the held-out test.csv and scores every
row, so the numbers are reproducible from your actual saved model.

Usage:
    python evaluate.py \
        --model Layer-2/models/phishing-model-3class \
        --test  Layer-2/data/dataset3/test.csv \
        --out   eval_report

Outputs land in the --out folder: PNGs + metrics.json + a short markdown summary.

Requires: torch, transformers, scikit-learn, matplotlib, seaborn, pandas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_label_map(model_dir: Path) -> dict:
    p = model_dir / "label_map.json"
    if p.exists():
        return {int(k): v for k, v in json.loads(p.read_text()).items()}
    return {0: "ham", 1: "phishing", 2: "ai_phish"}


def run_model_predictions(model_dir: Path, texts: list[str], max_length: int = 256):
    """Batch the test set through the model, return predicted label ids + probs."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()

    preds, all_probs = [], []
    batch = 16
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds.extend(probs.argmax(axis=1).tolist())
        all_probs.extend(probs.tolist())
        print(f"  scored {min(i+batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.array(preds), np.array(all_probs)


def plot_confusion(y_true, y_pred, label_names, out_path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    labels = sorted(label_names.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    names = [label_names[i] for i in labels]

    # Row-normalized so you can read recall per class off the diagonal
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="rocket_r",
                xticklabels=names, yticklabels=names, ax=ax,
                cbar_kws={"label": "row-normalized (recall)"},
                linewidths=0.5, linecolor="#2a333f")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Layer 2 (counts; shaded by recall)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class(y_true, y_pred, label_names, out_path: Path) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    labels = sorted(label_names.keys())
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    names = [label_names[i] for i in labels]

    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x - w, p, w, label="Precision", color="#39c5cf")
    ax.bar(x, r, w, label="Recall", color="#d9a441")
    ax.bar(x + w, f1, w, label="F1", color="#3fb950")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.2)
    for i, v in enumerate(f1):
        ax.text(i + w, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                       "f1": float(f1[i]), "support": int(support[i])}
            for i in range(len(names))}


def plot_layer_comparison(rules_f1, model_f1, fused_f1, out_path: Path):
    """The money chart: shows the hybrid beats either layer alone."""
    import matplotlib.pyplot as plt

    approaches = ["Layer 1\n(rules only)", "Layer 2\n(model only)", "Hybrid\n(fused)"]
    scores = [rules_f1, model_f1, fused_f1]
    colors = ["#8b98a5", "#39c5cf", "#f0506e"]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    bars = ax.bar(approaches, scores, color=colors, width=0.6)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Binary F1 (phishing vs. legit)")
    ax.set_title("Detection performance: each layer alone vs. combined")
    ax.grid(axis="y", alpha=0.2)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, s + 0.01, f"{s:.3f}",
                ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def binary_f1(y_true_bin, y_pred_bin) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true_bin, y_pred_bin, zero_division=0))


def main():
    ap = argparse.ArgumentParser(description="PhishShield evaluation report generator")
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True, help="test.csv with text,label columns")
    ap.add_argument("--out", default="eval_report")
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    model_dir = Path(args.model)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    label_names = load_label_map(model_dir)

    df = pd.read_csv(args.test)
    df = df[df["text"].notna()].reset_index(drop=True)
    y_true = df["label"].astype(int).to_numpy()
    texts = df["text"].astype(str).tolist()

    print(f"Evaluating {len(texts)} test emails through {model_dir.name}...")
    y_pred, probs = run_model_predictions(model_dir, texts, args.max_length)

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(y_true, y_pred)
    P, R, F1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    print(f"\nAccuracy: {acc:.4f} | Macro P: {P:.4f} R: {R:.4f} F1: {F1:.4f}")

    # 1 + 2: confusion matrix and per-class chart
    plot_confusion(y_true, y_pred, label_names, out / "confusion_matrix.png")
    per_class = plot_per_class(y_true, y_pred, label_names, out / "per_class_metrics.png")

    # 3: layer comparison — collapse to binary (phishing vs legit) for a fair 3-way compare.
    # Layer 2 (model): binary from the model's predictions.
    y_true_bin = (y_true > 0).astype(int)
    model_bin = (y_pred > 0).astype(int)
    model_f1 = binary_f1(y_true_bin, model_bin)

    # Layer 1 (rules): approximate — if the test set carries no header data, we
    # can't rerun Layer 1 per-row here, so we report the model vs fused and mark
    # rules as N/A unless a rules_pred column exists. If you want the true rules
    # number, add a 'rules_pred' column to test.csv (0/1) from layer1_detector.
    if "rules_pred" in df.columns:
        rules_f1 = binary_f1(y_true_bin, df["rules_pred"].astype(int).to_numpy())
    else:
        rules_f1 = None

    # Fused: OR-combine (either layer flags -> phishing), the recall-favoring rule
    if rules_f1 is not None:
        fused_bin = ((model_bin == 1) | (df["rules_pred"].astype(int).to_numpy() == 1)).astype(int)
        fused_f1 = binary_f1(y_true_bin, fused_bin)
        plot_layer_comparison(rules_f1, model_f1, fused_f1, out / "layer_comparison.png")
    else:
        fused_f1 = None

    # Save metrics json + a short markdown summary
    metrics = {
        "accuracy": float(acc),
        "macro": {"precision": float(P), "recall": float(R), "f1": float(F1)},
        "per_class": per_class,
        "binary_f1": {"model_only": model_f1, "rules_only": rules_f1, "fused": fused_f1},
        "n_test": len(texts),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    summary = [f"# Evaluation Report — {model_dir.name}\n",
               f"- Test emails: **{len(texts)}**",
               f"- Accuracy: **{acc:.4f}**",
               f"- Macro F1: **{F1:.4f}** (precision {P:.4f}, recall {R:.4f})\n",
               "## Per-class\n",
               "| class | precision | recall | f1 | support |",
               "|---|---|---|---|---|"]
    for name, m in per_class.items():
        summary.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} "
                       f"| {m['f1']:.3f} | {m['support']} |")
    summary.append("\n## Artifacts")
    summary.append("- `confusion_matrix.png`")
    summary.append("- `per_class_metrics.png`")
    if fused_f1 is not None:
        summary.append("- `layer_comparison.png`")
    else:
        summary.append("- (layer_comparison skipped — add a `rules_pred` column to test.csv "
                       "to include Layer 1 in the comparison)")
    (out / "REPORT.md").write_text("\n".join(summary))

    print(f"\nWrote report to {out.resolve()}/")
    print("  confusion_matrix.png, per_class_metrics.png, metrics.json, REPORT.md")
    if fused_f1 is not None:
        print("  layer_comparison.png")


if __name__ == "__main__":
    main()