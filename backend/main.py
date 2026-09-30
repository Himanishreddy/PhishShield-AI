"""
PhishShield AI — FastAPI Backend

Exposes the detection pipeline as a REST API so any client (web frontend,
mobile app, or a test tool like curl/Postman) can submit an email and get a
structured verdict back. This wraps the existing pipeline.run_pipeline — it
does NOT reimplement detection, so the API and the CLI/dashboard all share one
source of truth.

Run:
    pip install -r requirements.txt
    $env:OPENAI_API_KEY = "sk-..."            (PowerShell; needed for Layer 3)
    python -m uvicorn backend.main:app --port 8000

    (Use "python -m uvicorn", not "uvicorn": on Windows with Smart App Control
    the uvicorn.exe launcher is blocked, while python.exe is allowed.)

Then open the auto-generated interactive docs at:
    http://localhost:8000/docs      (Swagger UI — try requests in the browser)
    http://localhost:8000/redoc     (alternative docs view)

Endpoints:
    GET  /api/health            liveness + which model is loaded
    POST /api/analyze           analyze one raw email, return the verdict
    GET  /api/models            list available trained models

Project layout assumed (this file lives in backend/):
    Phishing/
      backend/main.py           <- this file
      pipeline.py
      Layer-1/layer1_detector.py
      Layer-2/predict.py
      Layer-2/models/...
      Layer-3/layer3_attribution.py   (AI adjudicator — final decision-maker)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Locate the project root (one level up from backend/) and load the modules
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required module not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod           # register before exec (Py 3.12+ dataclass)
    spec.loader.exec_module(mod)
    return mod


# Default model: prefer the 3-class model if present, else the binary one
def _default_model_dir() -> Path:
    models = ROOT / "Layer-2" / "models"
    for name in ("phishing-model-3class", "phishing-model"):
        p = models / name
        if p.exists():
            return p
    # fall back to the first directory found
    if models.exists():
        subdirs = [d for d in models.iterdir() if d.is_dir()]
        if subdirs:
            return subdirs[0]
    raise FileNotFoundError("No trained model found under Layer-2/models/")


# ---------------------------------------------------------------------------
# App + lazy-loaded detection system (load once, on first request)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(_app):
    """Load the model in the background right after the server starts, so the
    first real request doesn't wait for it and /api/health answers at once."""
    def _load():
        try:
            get_system()
        except Exception as e:                      # reported again on first request
            print(f"[startup] model preload failed: {e}", flush=True)
    threading.Thread(target=_load, daemon=True).start()
    yield


app = FastAPI(
    lifespan=_lifespan,
    title="PhishShield AI API",
    description="Hybrid phishing detection — rules + DistilBERT + AI adjudicator.",
    version="1.0.0",
)

# Which websites may call this API from a browser. Set ALLOWED_ORIGINS on the
# server to your frontend URL(s), comma-separated, e.g.
#   ALLOWED_ORIGINS=https://phishshield.vercel.app
# Unset = allow any origin (fine for local development). No cookies are used,
# so credentials are not needed.
_origins = [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class _System:
    """Holds the loaded modules + classifier. Populated on first use."""
    loaded = False
    layer1 = None
    predict = None
    pipeline = None
    layer3 = None
    classifier = None
    model_path: Optional[str] = None


SYS = _System()
_LOAD_LOCK = threading.Lock()   # two first requests must not load the model twice (RAM)


def _resolve_model(name: Optional[str]) -> Optional[str]:
    """Map a model NAME from a request to its folder under Layer-2/models/.
    Only plain folder names are accepted, never paths, so a caller cannot make
    the server load files from anywhere else on disk."""
    if not name:
        return None
    models_dir = (ROOT / "Layer-2" / "models").resolve()
    candidate = (models_dir / name).resolve()
    if (Path(name).name != name or candidate.parent != models_dir
            or not candidate.is_dir()):
        raise ValueError(f"Unknown model '{name}'. See GET /api/models.")
    return str(candidate)


def get_system(model_path: Optional[str] = None):
    """Load the detection system once and cache it on SYS."""
    with _LOAD_LOCK:
        return _get_system_locked(model_path)


def _get_system_locked(model_path: Optional[str] = None):
    target = model_path or (SYS.model_path or str(_default_model_dir()))

    # (Re)load only if not loaded yet or the requested model changed
    if SYS.loaded and target == SYS.model_path:
        return SYS

    SYS.layer1 = _load_module("layer1_detector", ROOT / "Layer-1" / "layer1_detector.py")
    SYS.predict = _load_module("predict", ROOT / "Layer-2" / "predict.py")
    SYS.pipeline = _load_module("pipeline", ROOT / "pipeline.py")

    l3_path = ROOT / "Layer-3" / "layer3_attribution.py"
    SYS.layer3 = _load_module("layer3_attribution", l3_path) if l3_path.exists() else None

    SYS.classifier = SYS.predict.PhishClassifier(target)
    SYS.model_path = target
    SYS.loaded = True
    return SYS



# ---------------------------------------------------------------------------
# Request / response schemas (Pydantic — gives validation + auto docs)
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    email: str = Field(..., description="Raw email text, including headers if available",
                       min_length=1, max_length=2_000_000)
    ensemble: bool = Field(False, description="Run Layer 2 on every email instead of gating")
    run_layer3: bool = Field(True, description="DEPRECATED and ignored: Layer 3 now runs on every email. Kept so existing clients that send this field keep working.")
    skip_layer3: bool = Field(False, description="Skip Layer 3 and return the Layer 1 + Layer 2 fused assessment only (for ablation/testing). Off by default.")
    model: Optional[str] = Field(None, description="Name of a model folder under Layer-2/models (optional; see GET /api/models)")

    model_config = {"json_schema_extra": {"examples": [{
        "email": "From: \"Microsoft Support\" <security-update@micros0ft-support.com>\n"
                 "Subject: URGENT: Account Suspended\n"
                 "Authentication-Results: mx; spf=fail; dkim=fail; dmarc=fail\n\n"
                 "Your account will be locked. Verify now: http://secure-login-portal.xyz",
        "ensemble": False}]}}


class HealthResponse(BaseModel):
    status: str
    model_loaded: Optional[str]
    layer3_available: bool
    layer3_api_key_set: bool = False   # True/False only; the key itself is never returned


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Liveness check + which model is currently loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=SYS.model_path,
        layer3_available=(ROOT / "Layer-3" / "layer3_attribution.py").exists(),
        layer3_api_key_set=bool(os.getenv("OPENAI_API_KEY")),
    )


@app.get("/api/models", tags=["meta"])
def list_models():
    """List trained models available under Layer-2/models/."""
    models_dir = ROOT / "Layer-2" / "models"
    if not models_dir.exists():
        return {"models": []}
    return {"models": [d.name for d in models_dir.iterdir() if d.is_dir()]}


@app.post("/api/analyze", tags=["detection"])
def analyze(req: AnalyzeRequest):
    """
    Analyze a single raw email and return the fused verdict plus each layer's
    evidence. This is the primary endpoint a frontend calls.
    """
    try:
        model_path = _resolve_model(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        system = get_system(model_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Layer 3 is the final authority and runs on every email. `run_layer3` from
    # older clients is ignored; only an explicit skip_layer3 bypasses it.
    layer3_mod = None if req.skip_layer3 else system.layer3

    try:
        result = system.pipeline.run_pipeline(
            req.email.encode("utf-8", errors="replace"),
            system.layer1,
            system.classifier,
            system.predict.load_eml_text,
            always_run_layer2=req.ensemble,
            layer3_mod=layer3_mod,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return result


@app.get("/", tags=["meta"])
def root():
    """Friendly landing pointer to the docs."""
    return {"message": "PhishShield AI API. Interactive docs at /docs"}
