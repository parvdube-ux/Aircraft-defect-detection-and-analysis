"""
EdgeVision FastAPI Backend
AI-Powered Aerospace Component Defect Detection & Maintenance Assistant
"""

import os
import json
import time
import datetime
import shutil
import base64
import tempfile
import subprocess
import sys
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import cv2
from PIL import Image
import torch
import gc
# Memory optimizations for cloud deployment (e.g. Render Free Tier)
torch.set_num_threads(1)
torch.set_grad_enabled(False)

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Optional heavy deps — loaded lazily so the API starts without GPU ──
_model = None
_class_names = None
_openai_client = None

# ── Config ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
WEIGHTS_PATH  = os.environ.get("EDGEVISION_WEIGHTS", str(BASE_DIR / "best.pt"))
HISTORY_FILE  = BASE_DIR / "inspection_history.json"
REPORTS_DIR   = BASE_DIR / "reports"
UPLOADS_DIR   = BASE_DIR / "uploads"
STATIC_DIR    = BASE_DIR / "static"

REPORTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")

OPENAI_MODEL    = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

HIGH_RISK_CLASSES = {"crack", "corrosion", "missing"}
RISK_COLORS_BGR = {
    "Low":      (0, 200, 0),
    "Medium":   (0, 165, 255),
    "High":     (0, 80, 255),
    "Critical": (0, 0, 255),
}
RISK_ORDER = ["Low", "Medium", "High", "Critical"]

SYSTEM_PROMPT = """You are EdgeVision's Maintenance Assistant, an aerospace MRO (maintenance, repair, overhaul) expert.
You translate computer-vision defect detections into clear, professional guidance for aircraft maintenance engineers.
Always:
- Summarize findings in plain language
- Map each defect to a maintenance recommendation and a preventive action
- Be conservative and flag anything uncertain for human inspection
- Note that the risk tags you are given are heuristic estimates, not a certified airworthiness determination
"""

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EdgeVision API",
    description="AI-Powered Aerospace Component Defect Detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")



# ── Pydantic models ────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    question: str
    session_id: str

class ConfigureOpenAI(BaseModel):
    api_key: str
    model: Optional[str] = "gpt-4o-mini"


# ── In-memory chat sessions ────────────────────────────────────────────
_chat_sessions: dict = {}   # session_id -> list of messages


# ── Helpers ────────────────────────────────────────────────────────────

def _get_model():
    global _model, _class_names
    if _model is None:
        if not os.path.exists(WEIGHTS_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model weights not found at '{WEIGHTS_PATH}'. "
                       "Upload best.pt or set EDGEVISION_WEIGHTS env var.",
            )
        from ultralytics import YOLO
        _model = YOLO(WEIGHTS_PATH)
        _class_names = _model.names  # dict {0: 'crack', ...}
    return _model, _class_names


def _get_openai():
    global _openai_client
    key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def estimate_risk(class_name: str, confidence: float, area_ratio: float) -> str:
    cls_lower = class_name.lower()
    score = 0
    score += 2 if any(k in cls_lower for k in HIGH_RISK_CLASSES) else 1
    if confidence >= 0.85:
        score += 2
    elif confidence >= 0.60:
        score += 1
    if area_ratio >= 0.15:
        score += 2
    elif area_ratio >= 0.05:
        score += 1
    if score >= 5: return "Critical"
    if score >= 4: return "High"
    if score >= 2: return "Medium"
    return "Low"


def img_to_b64(arr: np.ndarray) -> str:
    """Convert an RGB numpy array to a base64 PNG string."""
    success, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Image encoding failed")
    return base64.b64encode(buf.tobytes()).decode()


def build_inspection_context(detections: list, image_name: str) -> str:
    if not detections:
        return f"Inspection of '{image_name}': No defects detected above the confidence threshold."
    lines = [f"Inspection of '{image_name}' detected {len(detections)} potential defect(s):"]
    for i, d in enumerate(detections, 1):
        lines.append(
            f"{i}. {d['class']} — confidence {d['confidence']:.2f}, "
            f"relative size {d['area_ratio'] * 100:.1f}% of image, risk tag: {d['risk']}"
        )
    return "\n".join(lines)


def _load_history() -> list:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def _save_history(history: list):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


# ── Routes ─────────────────────────────────────────────────────────────

# The SPA root is served below using a catch-all route.


@app.get("/api/health")
async def health():
    model_ready = os.path.exists(WEIGHTS_PATH)
    openai_ready = bool(OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"))
    return {
        "status": "ok",
        "model_loaded": model_ready,
        "weights_path": WEIGHTS_PATH,
        "openai_configured": openai_ready,
        "timestamp": datetime.datetime.now().isoformat(),
    }


@app.post("/api/configure/openai")
async def configure_openai(cfg: ConfigureOpenAI):
    """Set the OpenAI API key + model at runtime (no restart required)."""
    global OPENAI_API_KEY, OPENAI_MODEL, _openai_client
    OPENAI_API_KEY = cfg.api_key
    OPENAI_MODEL   = cfg.model or "gpt-4o-mini"
    _openai_client = None          # force re-init
    os.environ["OPENAI_API_KEY"] = cfg.api_key
    return {"status": "ok", "model": OPENAI_MODEL}


# ── 1. Detect ───────────────────────────────────────────────────────────

@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, ge=0.01, le=0.99),
):
    """
    Run YOLOv8 defect detection + rule-based risk tagging on an uploaded image.
    Returns detections, annotated image (base64), and overall risk summary.
    """
    model, class_names = _get_model()

    # Save upload temporarily
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=UPLOADS_DIR) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        img_bgr = cv2.imread(tmp_path)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not read image.")
        h, w = img_bgr.shape[:2]
        img_area = h * w

        result = model.predict(tmp_path, conf=conf_threshold, verbose=False)[0]
        detections = []
        annotated  = img_bgr.copy()

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name   = class_names[cls_id]
            area_ratio = ((x2 - x1) * (y2 - y1)) / img_area
            risk = estimate_risk(cls_name, conf, area_ratio)
            detections.append({
                "class":      cls_name,
                "confidence": round(conf, 3),
                "bbox":       [x1, y1, x2, y2],
                "area_ratio": round(area_ratio, 4),
                "risk":       risk,
            })
            color = RISK_COLORS_BGR[risk]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, f"{cls_name} {conf:.2f} [{risk}]",
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        original_b64  = img_to_b64(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        annotated_b64 = img_to_b64(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        highest_risk = (
            max(detections, key=lambda d: RISK_ORDER.index(d["risk"]))["risk"]
            if detections else "None"
        )
        
        gc.collect()
        return {
            "image_name":    file.filename,
            "tmp_path":      tmp_path,
            "detections":    detections,
            "original_b64":  original_b64,
            "annotated_b64": annotated_b64,
            "highest_risk":  highest_risk,
            "total_defects": len(detections),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. EigenCAM ─────────────────────────────────────────────────────────

@app.post("/api/eigencam")
async def eigencam(
    file: UploadFile = File(...),
    img_size: int = Query(320),
    layer_index: int = Query(-2),
):
    """
    Generate an EigenCAM explainability heatmap for the uploaded image.
    Returns the heatmap overlay as base64.
    """
    model, _ = _get_model()

    # Clone YOLO-26-CAM if needed
    repo_dir = BASE_DIR / "yolo_cam_repo"
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/rigvedrs/YOLO-26-CAM.git", str(repo_dir)],
            check=True,
        )
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))

    try:
        from yolo_cam.eigen_cam import EigenCAM
        from yolo_cam.utils.image import show_cam_on_image
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"EigenCAM library not available: {e}")

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=UPLOADS_DIR) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        img_bgr = cv2.imread(tmp_path)
        img_bgr = cv2.resize(img_bgr, (img_size, img_size))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img_rgb) / 255.0

        target_layers = [model.model.model[layer_index]]
        cam = EigenCAM(model, target_layers, task="od")
        grayscale_cam_320 = cam(img_rgb_resized)[0, :, :]
        
        # Scale the heatmap back up to the ORIGINAL image size for a crystal clear output
        h, w = img_bgr.shape[:2]
        grayscale_cam_full = cv2.resize(grayscale_cam_320, (w, h))
        img_rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_float_full = np.float32(img_rgb_full) / 255.0
        
        cam_image = show_cam_on_image(img_float_full, grayscale_cam_full, use_rgb=True)

        gc.collect()
        return {
            "original_b64": img_to_b64(img_rgb_full),
            "heatmap_b64":  img_to_b64(cam_image),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. Maintenance Report (GenAI) ───────────────────────────────────────

@app.post("/api/report/generate")
async def generate_maintenance_report(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, ge=0.01, le=0.99),
):
    """
    Detect defects, then use OpenAI to generate a natural-language maintenance report.
    Also initialises a chat session keyed to this inspection.
    """
    client = _get_openai()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI not configured. POST /api/configure/openai first."
        )

    # Reuse detect logic inline
    model, class_names = _get_model()
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=UPLOADS_DIR) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        img_bgr = cv2.imread(tmp_path)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not read image.")
        img_bgr = cv2.resize(img_bgr, (320, 320))
        h, w = img_bgr.shape[:2]
        img_area = h * w
        result = model.predict(tmp_path, imgsz=320, conf=conf_threshold, verbose=False)[0]
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name   = class_names[cls_id]
            area_ratio = ((x2 - x1) * (y2 - y1)) / img_area
            risk = estimate_risk(cls_name, conf, area_ratio)
            detections.append({
                "class":      cls_name,
                "confidence": round(conf, 3),
                "bbox":       [x1, y1, x2, y2],
                "area_ratio": round(area_ratio, 4),
                "risk":       risk,
            })

        context = build_inspection_context(detections, file.filename)
        user_prompt = (
            f"{context}\n\n"
            "Please provide:\n"
            "1. A short executive summary (2–3 sentences)\n"
            "2. Risk assessment per defect\n"
            "3. Recommended maintenance actions (immediate vs. scheduled)\n"
            "4. Preventive measures to reduce recurrence\n"
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_prompt},
            ],
            temperature=0.3,
        )
        report_text = response.choices[0].message.content

        # Seed chat session
        session_id = f"session_{int(time.time() * 1000)}"
        _chat_sessions[session_id] = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Inspection data:\n{context}\nI may ask follow-up questions."},
            {"role": "assistant", "content": report_text},
        ]
        
        gc.collect()
        return {
            "report":      report_text,
            "detections":  detections,
            "session_id":  session_id,
            "image_name":  file.filename,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. Chat / Q&A ───────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(msg: ChatMessage):
    """Ask a follow-up question within an active inspection session."""
    client = _get_openai()
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI not configured.")

    history = _chat_sessions.get(msg.session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found. Generate a report first.")

    history.append({"role": "user", "content": msg.question})
    response = client.chat.completions.create(
        model=OPENAI_MODEL, messages=history, temperature=0.3
    )
    answer = response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    
    gc.collect()
    return {"answer": answer, "session_id": msg.session_id}


# ── 5. PDF Report ───────────────────────────────────────────────────────

@app.post("/api/pdf")
async def generate_pdf(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, ge=0.01, le=0.99),
):
    """
    Full pipeline: detect → EigenCAM → GenAI report → PDF.
    Returns the PDF filename for download.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise HTTPException(status_code=503, detail="fpdf2 not installed.")

    import re
    client = _get_openai()
    model, class_names = _get_model()

    # Ensure YOLO-CAM is available for PDF heatmap
    repo_dir = BASE_DIR / "yolo_cam_repo"
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/rigvedrs/YOLO-26-CAM.git", str(repo_dir)],
            check=True,
        )
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))
    try:
        from yolo_cam.eigen_cam import EigenCAM
        from yolo_cam.utils.image import show_cam_on_image
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"EigenCAM library not available: {e}")

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=UPLOADS_DIR) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        img_bgr = cv2.imread(tmp_path)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not read image.")
        img_bgr = cv2.resize(img_bgr, (320, 320))
        h, w = img_bgr.shape[:2]
        img_area = h * w
        result = model.predict(tmp_path, imgsz=320, conf=conf_threshold, verbose=False)[0]
        detections = []
        annotated  = img_bgr.copy()
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name   = class_names[cls_id]
            area_ratio = ((x2 - x1) * (y2 - y1)) / img_area
            risk = estimate_risk(cls_name, conf, area_ratio)
            detections.append({
                "class": cls_name, "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2], "area_ratio": round(area_ratio, 4), "risk": risk,
            })
            color = RISK_COLORS_BGR[risk]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # GenAI report text (if available)
        report_text = "OpenAI not configured — text report skipped."
        if client:
            context = build_inspection_context(detections, file.filename)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"{context}\nProvide a concise inspection report."},
                ],
                temperature=0.3,
            )
            report_text = resp.choices[0].message.content

        # Generate EigenCAM Heatmap
        img_size = 320
        layer_index = -2
        img_resized = cv2.resize(img_bgr, (img_size, img_size))
        img_rgb_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img_rgb_resized) / 255.0

        target_layers = [model.model.model[layer_index]]
        cam = EigenCAM(model, target_layers, task="od")
        grayscale_cam_320 = cam(img_rgb_resized)[0, :, :]
        
        # Scale the heatmap back up to the ORIGINAL image size
        h, w = img_bgr.shape[:2]
        grayscale_cam_full = cv2.resize(grayscale_cam_320, (w, h))
        img_rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_float_full = np.float32(img_rgb_full) / 255.0
        
        cam_image = show_cam_on_image(img_float_full, grayscale_cam_full, use_rgb=True)

        # Save images to tmp files for FPDF
        tmp_dir  = BASE_DIR / "tmp_report_imgs"
        tmp_dir.mkdir(exist_ok=True)
        orig_path = str(tmp_dir / "orig.png")
        ann_path  = str(tmp_dir / "annotated.png")
        heat_path = str(tmp_dir / "heatmap.png")
        
        Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(orig_path)
        Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(ann_path)
        Image.fromarray(cam_image).save(heat_path)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "EdgeVision - Aerospace Inspection Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Image: {file.filename}", ln=True)
        pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(4)
        
        # Display 3 images side by side (Original, Annotated, Heatmap)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Visual Inspection & EigenCAM Heatmap", ln=True)
        img_w = 60
        y0 = pdf.get_y()
        pdf.image(orig_path,  x=10,           y=y0, w=img_w)
        pdf.image(ann_path,   x=10 + img_w + 5, y=y0, w=img_w)
        pdf.image(heat_path,  x=10 + (img_w * 2) + 10, y=y0, w=img_w)
        pdf.ln(img_w * 0.75 + 10)
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Detected Defects", ln=True)
        pdf.set_font("Helvetica", "", 10)
        if detections:
            for d in detections:
                pdf.multi_cell(
                    0, 6,
                    f"- {d['class']} | conf {d['confidence']:.2f} | "
                    f"risk: {d['risk']} | size: {d['area_ratio'] * 100:.1f}%",
                )
        else:
            pdf.multi_cell(0, 6, "No defects detected above threshold.")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "AI Maintenance Assistant Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        
        # Remove markdown characters (*, #, _, -) and replace non-ASCII with spaces
        clean_text = re.sub(r'[*#_`]', '', report_text)
        clean_text = clean_text.replace('- ', '  ')
        clean_text = re.sub(r'[^\x00-\x7F]+', ' ', clean_text)
        
        pdf.multi_cell(0, 6, clean_text)

        pdf_name = f"inspection_{int(time.time())}.pdf"
        pdf_path = REPORTS_DIR / pdf_name
        pdf.output(str(pdf_path))

        # Log to history
        history = _load_history()
        history.append({
            "timestamp":  datetime.datetime.now().isoformat(),
            "image":      file.filename,
            "detections": detections,
            "summary":    report_text,
            "pdf_report": str(pdf_path),
        })
        _save_history(history)

        gc.collect()
        return {"pdf_url": f"/reports/{pdf_name}", "pdf_name": pdf_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 6. History ─────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history(limit: int = Query(20, ge=1, le=100)):
    """Return the N most-recent inspection records."""
    history = _load_history()
    risk_order = RISK_ORDER

    def highest_risk(detections):
        if not detections: return "None"
        return max(detections, key=lambda d: risk_order.index(d["risk"]))["risk"]

    records = [
        {
            "timestamp":    h["timestamp"],
            "image":        h["image"],
            "num_defects":  len(h["detections"]),
            "highest_risk": highest_risk(h["detections"]),
            "pdf_report":   h.get("pdf_report", ""),
        }
        for h in history[-limit:]
    ]
    return {"records": list(reversed(records)), "total": len(history)}


@app.delete("/api/history")
async def clear_history():
    _save_history([])
    return {"status": "cleared"}


# ── 7. Model info ──────────────────────────────────────────────────────

@app.get("/api/model/info")
async def model_info():
    model, class_names = _get_model()
    return {
        "weights": WEIGHTS_PATH,
        "classes": class_names,
        "num_classes": len(class_names),
        "device": str(next(model.model.parameters()).device),
    }


# ── Frontend SPA Catch-all Route ───────────────────────────────────────

# Mount assets from Vite build (Reload Triggered for env updates)
frontend_dist = BASE_DIR / "frontend" / "dist"
if frontend_dist.exists():
    # If the user accesses static assets directly
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Catch-all route to serve React Router SPA."""
    index_path = frontend_dist / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "Frontend not built yet. Run 'npm run build' in the frontend directory."}
