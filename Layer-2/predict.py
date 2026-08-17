"""
PhishShield AI — Layer 2 Inference / Prediction

Loads the fine-tuned DistilBERT model and scores email text for phishing.
Works on raw text, a .eml file, or piped stdin.

Usage:
    # Score a .eml file:
    python predict.py --model ./Layer-2/models/phishing-model --eml ./Layer-1/sample_phish.eml

    # Score a raw text string:
    python predict.py --model ./Layer-2/models/phishing-model --text "Your account is suspended, click to verify"

    # Score from stdin:
    echo "Urgent: confirm your password now" | python predict.py --model ./Layer-2/models/phishing-model

    # Score many .eml files in a folder:
    python predict.py --model ./Layer-2/models/phishing-model --dir ./some_emails

Outputs the predicted label, confidence, and full class probabilities as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_eml_text(path: Path) -> str:
    """Extract subject + body text from a .eml (mirrors dataset_builder)."""
    from email import message_from_bytes
    import re

    msg = message_from_bytes(path.read_bytes())
    subject = msg.get("Subject", "") or ""

    def strip_html(html: str) -> str:
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()

    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode(errors="ignore"))
            elif part.get_content_type() == "text/html" and not body_parts:
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(strip_html(payload.decode(errors="ignore")))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            text = payload.decode(errors="ignore")
            if msg.get_content_type() == "text/html":
                text = strip_html(text)
            body_parts.append(text)

    body = "\n".join(body_parts).strip()
    return f"{subject}\n\n{body}".strip() if subject else body


class PhishClassifier:
    """Wraps the fine-tuned model for repeated inference (load once, score many)."""

    def __init__(self, model_dir: str, max_length: int = 256):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

        # Load the human-readable label names if present
        label_map_path = Path(model_dir) / "label_map.json"
        if label_map_path.exists():
            raw = json.loads(label_map_path.read_text())
            self.label_names = {int(k): v for k, v in raw.items()}
        else:
            n = self.model.config.num_labels
            self.label_names = {i: f"class_{i}" for i in range(n)}

    def predict(self, text: str) -> dict:
        if not text.strip():
            return {"error": "empty text"}

        inputs = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1)[0].cpu().tolist()

        pred_id = int(max(range(len(probs)), key=lambda i: probs[i]))
        return {
            "predicted_label": self.label_names.get(pred_id, str(pred_id)),
            "confidence": round(probs[pred_id], 4),
            "probabilities": {
                self.label_names.get(i, str(i)): round(p, 4)
                for i, p in enumerate(probs)
            },
            "text_preview": text[:120] + ("…" if len(text) > 120 else ""),
        }


def main():
    ap = argparse.ArgumentParser(description="PhishShield Layer 2 inference")
    ap.add_argument("--model", required=True, help="Path to the fine-tuned model folder")
    ap.add_argument("--eml", help="Path to a single .eml file to score")
    ap.add_argument("--text", help="Raw text to score")
    ap.add_argument("--dir", help="Folder of .eml files to score in batch")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    clf = PhishClassifier(args.model, max_length=args.max_length)
    indent = 2 if args.pretty else None

    if args.dir:
        results = []
        for path in sorted(Path(args.dir).rglob("*.eml")):
            text = load_eml_text(path)
            r = clf.predict(text)
            r["file"] = path.name
            results.append(r)
        print(json.dumps(results, indent=indent))
        return

    if args.eml:
        text = load_eml_text(Path(args.eml))
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    print(json.dumps(clf.predict(text), indent=indent))


if __name__ == "__main__":
    main()