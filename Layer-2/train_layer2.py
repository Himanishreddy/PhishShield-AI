"""
PhishShield AI — Layer 2: DistilBERT Semantic Classifier (training)

Fine-tunes distilbert-base-uncased on the labeled corpus produced by
dataset_builder.py, and saves the trained model for inference.

Design choices that matter for a security classifier:
  * We track PRECISION, RECALL, and F1 — not raw accuracy. In phishing
    detection a false negative (a phish let through) is far worse than a
    false positive (a legit email flagged for review), so we care most
    about RECALL on the phishing class.
  * Memory-frugal defaults (small batch, short max_length, fp16 on GPU)
    so it has a fighting chance on a 2 GB laptop GPU. Bump them up on
    Colab or a bigger card.
  * Runs identically on CPU, a local CUDA GPU, or Colab — it auto-detects.

Two modes:
  --task binary    ham (0) vs. phishing (1)   [labels 1 & 2 collapsed to 1]
  --task multiclass ham (0) / phishing (1) / ai_phish (2)

Usage (local, GPU or CPU auto-detected):
    python train_layer2.py --data ./Layer-2/data/dataset --out ./Layer-2/models/distilbert-phish --task binary

    # Quick smoke test on the tiny demo dataset (1 epoch, tiny batch):
    python train_layer2.py --data ./dataset --out ./models/smoketest --task multiclass --epochs 1

Requires: torch, transformers, datasets, scikit-learn, accelerate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fine-tune DistilBERT for phishing detection")
    ap.add_argument("--data", required=True, help="Folder with train.csv / val.csv / test.csv")
    ap.add_argument("--out", required=True, help="Where to save the fine-tuned model")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary",
                    help="binary = ham vs phishing; multiclass = ham/phishing/ai_phish")
    ap.add_argument("--model", default="distilbert-base-uncased", help="Base model checkpoint")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Lower to 4 or 2 if you hit CUDA out-of-memory on a small GPU")
    ap.add_argument("--max-length", type=int, default=256,
                    help="Token cap per email. 256 is a good memory/coverage tradeoff; "
                         "drop to 128 to save memory, raise to 512 for long emails")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-fp16", action="store_true",
                    help="Disable mixed precision (fp16 is auto-used on GPU to save memory)")
    return ap


def load_split(data_dir: Path, name: str, task: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{name}.csv")
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    if task == "binary":
        # collapse phishing(1) and ai_phish(2) into a single "phishing"(1) class
        df["label"] = (df["label"] > 0).astype(int)
    return df.reset_index(drop=True)


def compute_metrics_builder(task: str):
    """Returns a compute_metrics fn that reports precision/recall/F1.

    For binary we report metrics on the positive (phishing) class explicitly,
    because that's the number that matters operationally. For multiclass we
    report macro averages plus per-class recall.
    """
    from sklearn.metrics import (
        precision_recall_fscore_support,
        accuracy_score,
        confusion_matrix,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)

        if task == "binary":
            p, r, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary", pos_label=1, zero_division=0
            )
            return {
                "accuracy": acc,
                "precision_phish": p,
                "recall_phish": r,       # <- the number to watch
                "f1_phish": f1,
            }
        else:
            p, r, f1, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )
            # per-class recall so you can see ai_phish recall specifically
            _, per_recall, _, _ = precision_recall_fscore_support(
                labels, preds, average=None, zero_division=0,
                labels=[0, 1, 2],
            )
            return {
                "accuracy": acc,
                "precision_macro": p,
                "recall_macro": r,
                "f1_macro": f1,
                "recall_ham": per_recall[0],
                "recall_phishing": per_recall[1],
                "recall_ai_phish": per_recall[2],
            }

    return compute_metrics


def main():
    args = build_argparser().parse_args()

    # Imports deferred so --help works without the heavy deps installed
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        set_seed,
    )

    set_seed(args.seed)
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (device == "cuda") and not args.no_fp16
    num_labels = 2 if args.task == "binary" else 3

    print(f"Device: {device} | task: {args.task} | labels: {num_labels} | fp16: {use_fp16}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"GPU: {props.name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 3 and args.batch_size > 8:
            print("  ! Low VRAM detected — if you hit OOM, rerun with --batch-size 4 "
                  "and/or --max-length 128")

    # ---- Load data ----
    train_df = load_split(data_dir, "train", args.task)
    val_df = load_split(data_dir, "val", args.task)
    test_df = load_split(data_dir, "test", args.task)
    print(f"Loaded: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # ---- Tokenize ----
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ---- Model ----
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    # ---- Class weights: penalize missing phishing more than false alarms ----
    # Higher weight on positive/phish classes nudges the model toward recall.
    label_counts = train_df["label"].value_counts().sort_index()
    total = label_counts.sum()
    weights = torch.tensor(
        [total / (num_labels * label_counts.get(i, 1)) for i in range(num_labels)],
        dtype=torch.float,
    )
    print(f"Class weights (recall-favoring): {weights.tolist()}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights.to(model.device))
            loss = loss_fct(
                outputs.logits.view(-1, num_labels), labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    # ---- Training args ----
    # Pick the metric to select the best checkpoint on: phishing recall.
    best_metric = "recall_phish" if args.task == "binary" else "f1_macro"

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_builder(args.task),
    )

    # ---- Train ----
    print("\nTraining...")
    trainer.train()

    # ---- Evaluate on held-out test set ----
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_ds)
    print(json.dumps(test_metrics, indent=2, default=float))

    # ---- Save final model + tokenizer + label map ----
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    label_map = ({0: "ham", 1: "phishing"} if args.task == "binary"
                 else {0: "ham", 1: "phishing", 2: "ai_phish"})
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    print(f"\nSaved fine-tuned model to {out_dir.resolve()}")
    print("Next: use this model in the inference/pipeline step to score "
          "emails that Layer 1 escalates.")


if __name__ == "__main__":
    main()