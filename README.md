# PhishShield AI

A hybrid, layered phishing-detection pipeline that combines fast deterministic
rules, a fine-tuned transformer classifier, and an AI adjudicator that makes the
final call — surfaced through a React web app, a FastAPI backend and a
Security Operations Center (SOC) triage dashboard.

The project's focus is detecting **AI-generated phishing** alongside classic
human-written phishing, and giving a security analyst not just a verdict but the
*evidence* behind it.

**Live demo:** <https://phish-shield-ai-omega.vercel.app> — backend on Render's free tier,
so the first check after a period of inactivity can take about a minute while it wakes up.

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
                 │ (every email)
                 ▼
  ┌──────────────────────────────┐
  │ Layer 3 — AI adjudicator      │  final verdict + plain-
  │ weighs all evidence, with     │  language explanation
  │ code-enforced guardrails      │  (OpenAI API)
  └──────────────┬───────────────┘
                 ▼
     React app / FastAPI / Streamlit
```

The two detection layers cover each other's blind spots. Layer 1 reads headers,
so it catches a spoofed sender or failed authentication that a text model can't
see. Layer 2 reads language, so it catches a fluent, well-crafted email sent
from an authenticated domain that the rules would wave through. Neither alone is
sufficient; together they are complementary.

---

## Results

Layer 2 (the current model, **V5**) was fine-tuned from `distilbert-base-uncased`
on 21,776 training emails — 12,161 legitimate, 5,754 human-written phishing and
3,861 AI-generated phishing — drawn from public corpora (a phishing/legitimate
email set, an AI-generated phishing set, SpamAssassin, Nazario 2022–2025, Enron)
plus a small number of the author's own anonymised inbox emails. The splits are
built with near-duplicate clustering, so an email template never appears in
more than one of train / validation / test (leakage check passed).

### Internal test set (4,371 emails, same sources as training)

| Metric | Score |
|---|---|
| Accuracy | 98.72% |
| Macro precision | 98.80% |
| Macro recall | 98.71% |
| Macro F1 | 98.76% |
| Recall — legitimate | 98.87% |
| Recall — human phishing | 97.27% |
| Recall — AI phishing | 100% |

### External benchmark — 500 unseen legitimate Enron emails

None of these 500 emails appear in the training data (checked for exact and
near-duplicate matches).

| Metric | Score |
|---|---|
| Correctly classified as legitimate | 491 / 500 |
| False-positive rate | **1.80%** |
| Mean P(legitimate) | 98.26% |

Result files: `results/v5_test/metrics.json`, `results/v5_enron/ood_report.json`.

Metrics favor **recall** on the phishing classes by design: in security, a
missed phish (false negative) is more costly than a false alarm (false
positive), so the training loss is class-weighted accordingly.

### Development history

The first model flagged **15 of 20 (75%)** real legitimate emails from the
author's inbox as phishing — security notices, OTP mails, campus recruitment
and institutional circulars. The cause was that security/account vocabulary
appeared almost only in the phishing class. Later versions fixed this by adding
legitimate "hard negatives" (SpamAssassin hard ham, real inbox mail, Enron).
Because 16 of those 20 inbox emails were then added to training, they are no
longer a held-out test; the Enron benchmark above is the held-out measure.

### Honest limitations

These numbers describe performance **on this dataset's distribution**, and they
should be read with these caveats — discussed here rather than hidden, because
understanding them is part of the engineering:

1. **Source-separation effect.** The perfect AI-phishing recall partly reflects
   the model learning artifacts that distinguish the *specific corpora* used
   (formatting, length, collection method), not a general notion of
   "AI-written-ness." Honest framing: the model separates *these datasets*
   near-perfectly; a production-grade "AI detector" would need harder
   negatives and mixed-source validation. (Listed under Future Work.)

2. **Layer 3 is an AI judgement, not proof.** The adjudicator can be wrong, so
   it is constrained in code: it cannot clear an email that Layer 1 has strong
   technical evidence against, cannot clear a header-less email that Layer 2
   flagged, and can only use the "AI-assisted" label when Layer 2 predicted
   `ai_phish`. Before the model sees anything, code computes verified facts
   (DMARC alignment, mailing-list/ARC forwarding, Reply-To relation, per-link
   risk features, requests for passwords/OTPs/payments) so the model is not left
   to interpret raw headers. The email itself is fenced as untrusted data to
   resist prompt injection. If the API is unavailable, the result falls back to
   the Layer 1 + Layer 2 assessment and is clearly marked as degraded.

3. **Indian financial / transactional mail is still a weak spot.** Of the 4
   real inbox emails that were never used in training, V5 flagged 3 as phishing
   (an income-tax e-verification confirmation, a GST invoice and a merchandise
   sale notice). The sample is small, but it shows where more legitimate
   training data is needed.

4. **No year-held-out phishing benchmark.** Nazario 2022–2025 phishing emails
   are part of the training/validation/test pool, so phishing detection is
   measured on the internal test set only, not on an unseen year.

---

## Project layout

```
Phishing/
├── pipeline.py               Orchestrator — runs all layers, emits one verdict
├── soc_dashboard.py          Streamlit SOC triage UI
├── test_layer3.py            Offline tests for Layer 3 + pipeline (no API key needed)
├── backend/main.py           FastAPI REST API (POST /api/analyze)
├── frontend/                 React (Vite) web app
├── requirements.txt          Everything (training, dashboard, evaluation, API)
├── requirements-api.txt      Only what the deployed API needs (used by Dockerfile)
├── Dockerfile                Backend container for Render
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
    └── layer3_attribution.py AI adjudicator (final verdict, guardrails)
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

### 4. Layer 3 API key

Layer 3 calls the OpenAI API. Set the key as an environment variable (never in
code):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Without a key the system still runs; results fall back to Layer 1 + Layer 2 and
are marked as degraded.

---

## Usage

### Command line (full pipeline)

```bash
python pipeline.py \
    --model Layer-2/models/phishing-model-3class \
    --eml Layer-1/sample_phish.eml \
    --pretty
```

Outputs one JSON result: Layer 3's final verdict (`final`), the pre-Layer-3
fused assessment (`fusion`), and each layer's evidence.

Useful flags:
- `--always-run-layer2` — ensemble mode: run the classifier on every email
  instead of gating it behind Layer 1.
- `--dir <folder>` — score every `.eml` in a folder.
- `--no-layer3` — Layers 1 + 2 + fusion only (ablation / offline).

### Tests

```bash
python test_layer3.py
```

### Web app

```bash
python -m uvicorn backend.main:app --port 8000     # backend
cd frontend && npm install && npm run dev          # frontend, http://localhost:5173
```

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
recall. The fused result is passed to Layer 3 as evidence; Layer 3 runs on every
email and issues the final verdict.

---

## Future work

- Harder negatives and cross-source validation to move Layer 2 from
  "dataset separation" toward genuine AI-generated-text detection.
- Live WHOIS/RDAP integration for real domain-age signals in Layer 1
  (currently stubbed behind a clean interface).
- Explainability overlays (token attributions) in the dashboard.
- Feedback loop: analyst corrections of Layer 3 verdicts as future training signal.

---

## Acknowledgements
 Uses `distilbert-base-uncased` (Hugging Face
Transformers), the OpenAI API for the Layer 3 adjudicator, FastAPI, React and
Streamlit.
Datasets are public phishing/legitimate email corpora; see Setup for sourcing.
