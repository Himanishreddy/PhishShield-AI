# PhishShield AI

A hybrid, layered phishing-detection pipeline that combines fast deterministic
rules, a fine-tuned transformer classifier, and a local LLM attribution layer —
surfaced through a Security Operations Center (SOC) triage dashboard.

The project's focus is detecting **AI-generated phishing** alongside classic
human-written phishing, and giving a security analyst not just a verdict but the
*evidence* behind it.

---

## Why a layered design

No single technique catches every phishing email, and the cheap techniques miss
different things than the expensive ones. PhishShield runs a funnel: each stage
is more costly than the last, so most email is resolved early and only the
suspicious minority reaches the heavy analysis.

```
  Incoming email
        │
        ▼
  ┌──────────────────────────────┐
  │ Layer 1 — Rules & headers    │  fast, free, deterministic
  │ SPF/DKIM/DMARC, lookalike     │
  │ domains, urgency, link checks │
  └──────────────┬───────────────┘
                 │ risk score + verdict
                 ▼
  ┌──────────────────────────────┐
  │ Layer 2 — DistilBERT          │  reads the language
  │ ham / phishing / ai_phish     │
  └──────────────┬───────────────┘
                 │ class probabilities
                 ▼
  ┌──────────────────────────────┐
  │ Fusion — combined verdict     │  recall-favoring
  └──────────────┬───────────────┘
                 │ (only if phishing)
                 ▼
  ┌──────────────────────────────┐
  │ Layer 3 — LLM attribution     │  inferred threat intel
  │ objective, triggers, persona  │  (local, via Ollama)
  └──────────────┬───────────────┘
                 ▼
        SOC Dashboard (Streamlit)
```

The two detection layers cover each other's blind spots. Layer 1 reads headers,
so it catches a spoofed sender or failed authentication that a text model can't
see. Layer 2 reads language, so it catches a fluent, well-crafted email sent
from an authenticated domain that the rules would wave through. Neither alone is
sufficient; together they are complementary.

---

## Results

Layer 2 was fine-tuned from `distilbert-base-uncased` on a 3-class corpus of
23,522 emails (legitimate / human-phishing / AI-generated-phishing), assembled
from public datasets. On a held-out test set:

| Metric | Score |
|---|---|
| Macro F1 | 99.2% |
| Recall — legitimate | 99.3% |
| Recall — human phishing | 98.3% |
| Recall — AI phishing | 100% |

Metrics favor **recall** on the phishing classes by design: in security, a
missed phish (false negative) is more costly than a false alarm (false
positive), so the training loss is class-weighted accordingly.

### Honest limitations

These numbers describe performance **on this dataset's distribution**, and they
should be read with two caveats — both of which are discussed here rather than
hidden, because understanding them is part of the engineering:

1. **Source-separation effect.** The perfect AI-phishing recall partly reflects
   the model learning artifacts that distinguish the *specific corpora* used
   (formatting, length, collection method), not a general notion of
   "AI-written-ness." An out-of-distribution test — a hand-crafted AI-style
   email from neither source — was misclassified, which exposes this. Honest
   framing: the model separates *these datasets* near-perfectly; a
   production-grade "AI detector" would need harder negatives and mixed-source
   validation. (Listed under Future Work.)

2. **Layer 3 is inference, not forensics.** The attribution layer does **not**
   recover an attacker's real prompt or identity — an LLM cannot reconstruct
   another model's input. It produces a *plausible hypothesis* about
   methodology to accelerate analyst triage. Every Layer 3 output carries this
   disclaimer, and its "illustrative generation prompt" is explicitly an
   example, not a recovered artifact. In testing it sometimes mislabels the
   objective — which is exactly why it is presented as an analyst aid to be
   verified, never as ground truth.

---

## Project layout

```
Phishing/
├── pipeline.py               Orchestrator — runs all layers, emits one verdict
├── soc_dashboard.py          Streamlit SOC triage UI
├── requirements.txt
├── README.md
│
├── Layer-1/
│   └── layer1_detector.py    Deterministic rules / header analysis
│
├── Layer-2/
│   ├── dataset_builder.py    Assemble labeled train/val/test splits
│   ├── prep_dataset.py       Normalize any raw dataset -> text,label CSV
│   ├── train_layer2.py       Fine-tune DistilBERT (binary or 3-class)
│   ├── predict.py            Inference wrapper for the trained model
│   ├── data/                 (git-ignored) datasets + built splits
│   └── models/               (git-ignored) trained model(s)
│
└── Layer-3/
    └── layer3_attribution.py LLM attribution via local Ollama
```

Models and datasets are intentionally **not** in the repo (they're large and
regenerable). Rebuild them by following Setup below.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration (optional), install `torch` from
<https://pytorch.org/get-started/locally/> first. CPU works fine otherwise.

### 2. Build the dataset (to train Layer 2)

Download public phishing corpora — e.g. a human-phishing + legitimate set and an
LLM-generated set — then normalize and combine them:

```bash
# Inspect a raw download to see its columns
python Layer-2/prep_dataset.py --in raw.csv --inspect

# Normalize into text,label form (labels: 0=ham, 1=phishing, 2=ai_phish)
python Layer-2/prep_dataset.py --in raw.csv --out clean.csv \
    --text-col "Email Text" --label-col "Email Type" \
    --map "Safe Email=0,Phishing Email=1"

# Build stratified train/val/test splits
python Layer-2/dataset_builder.py --csv clean.csv --out Layer-2/data/dataset3
```

### 3. Train Layer 2

```bash
python Layer-2/train_layer2.py \
    --data Layer-2/data/dataset3 \
    --out Layer-2/models/phishing-model-3class \
    --task multiclass --epochs 3
```

### 4. Install Ollama (for Layer 3)

```bash
# Install from https://ollama.com, then:
ollama pull llama3.2
```

Layer 3 is optional — the pipeline and dashboard run without it and degrade
gracefully if Ollama isn't available.

---

## Usage

### Command line (full pipeline)

```bash
python pipeline.py \
    --model Layer-2/models/phishing-model-3class \
    --eml Layer-1/sample_phish.eml \
    --layer3 --pretty
```

Outputs one combined JSON verdict: the fused risk score, each layer's evidence,
and — on confirmed phishing with `--layer3` — the attribution analysis.

Useful flags:
- `--always-run-layer2` — ensemble mode: run the classifier on every email
  instead of gating it behind Layer 1.
- `--dir <folder>` — score every `.eml` in a folder.

### Dashboard

```bash
python -m streamlit run soc_dashboard.py
```

Paste an email (headers included), hit **Analyze**, and read the verdict banner,
per-layer evidence panels, and — with the sidebar toggle on — the Layer 3
attribution. The sidebar auto-discovers trained models and lets you switch
between them.

---

## How the fusion works

Layer 1 runs on every email. Layer 2 runs unless Layer 1 is confident the email
is clean *and* the sender passed authentication (a compute-saving gate; toggle
`--always-run-layer2` to disable it). The two signals are blended into a single
0–100 score, but either layer can raise an alert on its own — the fusion favors
recall. Layer 3 runs only when the fused verdict is `phishing`.

---

## Future work

- Harder negatives and cross-source validation to move Layer 2 from
  "dataset separation" toward genuine AI-generated-text detection.
- Live WHOIS/RDAP integration for real domain-age signals in Layer 1
  (currently stubbed behind a clean interface).
- Explainability overlays (token attributions) in the dashboard.
- Feedback loop: analyst verdicts on Layer 3 output as future training signal.

---

## Acknowledgements
 Uses `distilbert-base-uncased` (Hugging Face
Transformers), Ollama for local LLM inference, and Streamlit for the dashboard.
Datasets are public phishing/legitimate email corpora; see Setup for sourcing.
