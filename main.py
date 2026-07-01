import os
import json
import time
import datetime
import shutil
import base64
import tempfile
import subprocess
import sys
import re
from typing import Optional
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
import gc
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Thread & gradient optimizations for CPU deployment
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# ── Config ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
WEIGHTS_PATH  = os.environ.get("EDGEVISION_WEIGHTS", str(BASE_DIR / "best.pt"))
HISTORY_FILE  = BASE_DIR / "inspection_history.json"
REPORTS_DIR   = BASE_DIR / "reports"
UPLOADS_DIR   = BASE_DIR / "uploads"
YOLO_CAM_DIR  = BASE_DIR / "yolo_cam_repo"

REPORTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

HIGH_RISK_CLASSES = {"crack", "corrosion", "missing"}
RISK_COLORS_BGR = {
    "Low":      (0, 200, 0),
    "Medium":   (0, 165, 255),
    "High":     (0, 80, 255),
    "Critical": (0, 0, 255),
}
RISK_COLORS_HEX = {
    "Low":      "#10b981",
    "Medium":   "#f59e0b",
    "High":     "#ef4444",
    "Critical": "#dc2626",
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

# ── Page Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="EdgeVision AI v2.2 - Aerospace Inspection Hub",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Premium Dark Theme ───────────────────────────────────
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Base Page Setup */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #050505 !important;
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(14, 165, 233, 0.12) 0px, transparent 50%) !important;
            background-attachment: fixed !important;
            color: #ffffff;
        }

        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0c0c0d !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        /* Glass Cards Container */
        .glass-card {
            background: rgba(17, 17, 17, 0.7);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
        }

        .glass-header {
            font-family: 'Outfit', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
            color: #ffffff;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 8px;
        }

        /* Badge Styles */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
        }
        .badge-none { background: rgba(255, 255, 255, 0.1); color: #9ca3af; border: 1px solid rgba(255, 255, 255, 0.2); }
        .badge-low { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-medium { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-high { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
        .badge-critical { 
            background: rgba(220, 38, 38, 0.2); 
            color: #fca5a5; 
            border: 1px solid #ef4444;
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
        }

        /* Metrics Display */
        .metric-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }
        .metric-val {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 4px;
        }

        /* Defect Items */
        .defect-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .defect-info {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .defect-title {
            font-weight: 600;
            text-transform: capitalize;
            font-size: 0.95rem;
            color: #ffffff;
        }
        .defect-meta {
            font-size: 0.8rem;
            color: #9ca3af;
        }

        /* Alert Banner */
        .alert-banner {
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
            border-left: 4px solid #ef4444;
            border-radius: 0 10px 10px 0;
            padding: 16px;
            margin-top: 16px;
        }
        .alert-header {
            font-weight: 700;
            color: #fca5a5;
            font-size: 0.95rem;
            margin-bottom: 4px;
        }
        .alert-text {
            color: rgba(255, 255, 255, 0.85);
            font-size: 0.85rem;
            line-height: 1.4;
        }

        /* Tech Report Styles */
        .report-section {
            background: rgba(14, 165, 233, 0.03);
            border: 1px solid rgba(14, 165, 233, 0.15);
            border-radius: 12px;
            padding: 20px;
            margin-top: 16px;
            position: relative;
        }
        .report-section::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 3px;
            background: linear-gradient(90deg, #3b82f6, #0ea5e9);
            border-radius: 12px 12px 0 0;
        }

        /* Logo styling */
        .logo-wrap {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        .logo-box {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #3b82f6, #0ea5e9);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
        }

        /* Chat bubbles override */
        .stChatMessage {
            background-color: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ── Model & Dependency Management ───────────────────────────────────────

@st.cache_resource
def load_yolo_model(weights_path: str):
    if not os.path.exists(weights_path):
        return None, f"Model weights not found at '{weights_path}'."
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        return model, None
    except Exception as e:
        return None, f"Failed to load YOLO model: {e}"

def load_openai_client(api_key: str):
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"Error initializing OpenAI: {e}")
        return None

def clone_yolo_cam():
    if not YOLO_CAM_DIR.exists():
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/rigvedrs/YOLO-26-CAM.git", str(YOLO_CAM_DIR)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            return False, f"Failed to clone EigenCAM repository: {e}"
    if str(YOLO_CAM_DIR) not in sys.path:
        sys.path.append(str(YOLO_CAM_DIR))
    return True, None

# ── Helper Functions ────────────────────────────────────────────────────

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

def load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history: list):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Failed to save inspection history: {e}")

# ── App Layout & Sidebar ────────────────────────────────────────────────

# Top Logo / Title
st.markdown("""
    <div class="logo-wrap">
        <div class="logo-box">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h4l2-9 5 18 3-9h6"/></svg>
        </div>
        <span style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 700; letter-spacing: -0.5px;">EdgeVision AI v2.2</span>
        <span style="color: #9ca3af; font-size: 0.85rem; margin-top: 5px;">| Aerospace MRO Assistant</span>
    </div>
""", unsafe_allow_html=True)

# Session State Initialization
if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "vision_report" not in st.session_state:
    st.session_state.vision_report = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("OPENAI_API_KEY", "")
if "gpt_model" not in st.session_state:
    st.session_state.gpt_model = "gpt-4o-mini"
if "acknowledged_escalation" not in st.session_state:
    st.session_state.acknowledged_escalation = False

# Sidebar Configuration
st.sidebar.title("Configuration")

# OpenAI API Key Input
openai_key_input = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=st.session_state.api_key,
    placeholder="sk-...",
    help="Enter your OpenAI API key to enable GPT analysis and MRO Assistant chat."
)
if openai_key_input != st.session_state.api_key:
    st.session_state.api_key = openai_key_input
    os.environ["OPENAI_API_KEY"] = openai_key_input
    # Clear client cache by changing states
    if "openai_client" in st.session_state:
        del st.session_state.openai_client

st.session_state.openai_configured = bool(st.session_state.api_key)

# Model Settings
model_options = ["gpt-4o-mini", "gpt-4"]
gpt_model_select = st.sidebar.selectbox("GPT Model", model_options, index=model_options.index(st.session_state.gpt_model))
if gpt_model_select != st.session_state.gpt_model:
    st.session_state.gpt_model = gpt_model_select

# Confidence Threshold
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.01,
    max_value=0.99,
    value=0.25,
    step=0.01,
    help="Defects with confidence scores below this value will be ignored by YOLO."
)

st.sidebar.markdown("---")

# History Section
st.sidebar.title("Inspection History")
history_data = load_history()

if not history_data:
    st.sidebar.info("No inspection records found.")
else:
    # Clear history option
    if st.sidebar.button("Clear History Log", use_container_width=True):
        save_history([])
        st.sidebar.success("History cleared!")
        st.rerun()

    # Display list of past reports
    st.sidebar.markdown("**Recent Inspections**")
    for idx, item in enumerate(reversed(history_data[-15:])):
        # Construct summary label
        time_str = datetime.datetime.fromisoformat(item["timestamp"]).strftime("%m/%d %H:%M")
        img_name = item["image"]
        num_defects = len(item["detections"])
        highest_risk = "None"
        if num_defects > 0:
            highest_risk = max(item["detections"], key=lambda d: RISK_ORDER.index(d["risk"]))["risk"]

        btn_label = f"📁 {time_str} - {img_name[:12]} ({num_defects} Defect, {highest_risk})"
        if st.sidebar.button(btn_label, key=f"hist_{idx}", use_container_width=True):
            # Load selected history item into session state
            st.session_state.current_file_hash = f"history_{item['timestamp']}"
            
            # Load images if available or set to dummy
            st.session_state.analysis_results = {
                "image_name": item["image"],
                "detections": item["detections"],
                "original_img": None, # Will remain None or loaded from uploads if it exists
                "annotated_img": None,
                "heatmap_img": None,
                "highest_risk": highest_risk,
                "total_defects": num_defects,
                "report": item.get("summary", "No technical report saved."),
                "pdf_url": item.get("pdf_report", "")
            }
            st.session_state.chat_history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Loaded historical inspection data."},
                {"role": "assistant", "content": item.get("summary", "")}
            ]
            st.session_state.session_id = f"session_hist_{idx}"
            st.session_state.acknowledged_escalation = False

# Load YOLO model
model, load_err = load_yolo_model(WEIGHTS_PATH)
if load_err:
    st.error(load_err)
    st.stop()

# ── Main Content Area ───────────────────────────────────────────────────

# Layout columns for Dashboard vs Explanations
tab_dashboard, tab_cam, tab_chat = st.tabs(["🔍 Inspection Dashboard", "🔬 Explainability CAM", "💬 AI Assistant Chat"])

# File Uploader
uploaded_file = st.file_uploader("Upload Component Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Compute unique hash to prevent duplicate runs
    file_bytes = uploaded_file.getvalue()
    file_hash = str(hash(file_bytes)) + f"_{conf_threshold}"
    
    if st.session_state.current_file_hash != file_hash:
        st.session_state.current_file_hash = file_hash
        st.session_state.acknowledged_escalation = False
        st.session_state.chat_history = []
        
        # Save temp file for YOLO
        suffix = Path(uploaded_file.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=UPLOADS_DIR) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # Load OpenCV image
            img_bgr = cv2.imread(tmp_path)
            if img_bgr is None:
                st.error("Failed to read the uploaded image.")
                st.stop()
            h, w = img_bgr.shape[:2]
            img_area = h * w

            # YOLO inference
            result = model.predict(tmp_path, conf=conf_threshold, verbose=False)[0]
            detections = []
            annotated = img_bgr.copy()
            class_names = model.names

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = class_names[cls_id]
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

            # Convert images to PIL for display
            orig_pil = Image.open(tmp_path)
            ann_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            
            highest_risk = (
                max(detections, key=lambda d: RISK_ORDER.index(d["risk"]))["risk"]
                if detections else "None"
            )

            # Generate EigenCAM
            heatmap_pil = None
            cam_ok, cam_err = clone_yolo_cam()
            if cam_ok:
                try:
                    from yolo_cam.eigen_cam import EigenCAM
                    from yolo_cam.utils.image import show_cam_on_image
                    
                    img_resized = cv2.resize(img_bgr, (320, 320))
                    img_rgb_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    
                    target_layers = [model.model.model[-2]]
                    cam = EigenCAM(model, target_layers, task="od")
                    grayscale_cam_320 = cam(img_rgb_resized)[0, :, :]
                    
                    grayscale_cam_full = cv2.resize(grayscale_cam_320, (w, h))
                    img_rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_float_full = np.float32(img_rgb_full) / 255.0
                    
                    cam_image = show_cam_on_image(img_float_full, grayscale_cam_full, use_rgb=True)
                    heatmap_pil = Image.fromarray(cam_image)
                except Exception as e:
                    cam_err = f"EigenCAM error: {e}"
            
            # Generate OpenAI Maintenance Report if key is provided
            openai_report = "OpenAI API key not configured. Enable in sidebar to generate AI reports."
            session_id = f"session_{int(time.time() * 1000)}"
            vision_analysis = "OpenAI API key not configured. Direct Vision Analysis skipped."
            
            client = load_openai_client(st.session_state.api_key)
            if client:
                try:
                    # 1. Standard Technical Report
                    context = build_inspection_context(detections, uploaded_file.name)
                    user_prompt = (
                        f"{context}\n\n"
                        "Please provide:\n"
                        "1. A short executive summary (2–3 sentences)\n"
                        "2. Risk assessment per defect\n"
                        "3. Recommended maintenance actions (immediate vs. scheduled)\n"
                        "4. Preventive measures to reduce recurrence\n"
                    )

                    response = client.chat.completions.create(
                        model=st.session_state.gpt_model,
                        messages=[
                            {"role": "system",  "content": SYSTEM_PROMPT},
                            {"role": "user",    "content": user_prompt},
                        ],
                        temperature=0.3,
                    )
                    openai_report = response.choices[0].message.content
                    
                    # Seed chat session
                    st.session_state.chat_history = [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": f"Inspection data:\n{context}\nI may ask follow-up questions."},
                        {"role": "assistant", "content": openai_report},
                    ]
                    st.session_state.session_id = session_id

                    # 2. Vision API direct call
                    # Convert original to base64
                    success, buf = cv2.imencode(".png", img_bgr)
                    if success:
                        img_b64 = base64.b64encode(buf.tobytes()).decode()
                        vis_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an expert aerospace maintenance vision AI. Analyze the image and provide a highly accurate assessment of any visible defects. Format your response in clean Markdown with bolded headers and bullet points."
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        { "type": "text", "text": "Please analyze this component image. What defects do you see? Provide a detailed visual inspection summary." },
                                        { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{img_b64}" } }
                                    ]
                                }
                            ],
                            max_tokens=400,
                            temperature=0.2
                        )
                        vision_analysis = vis_response.choices[0].message.content
                except Exception as e:
                    openai_report = f"AI API Error: {e}"
                    vision_analysis = f"AI Vision Error: {e}"

            # Save report PDF using FPDF
            pdf_path = ""
            try:
                from fpdf import FPDF
                
                # Save visual images for PDF generator
                tmp_dir = BASE_DIR / "tmp_report_imgs"
                tmp_dir.mkdir(exist_ok=True)
                orig_path = str(tmp_dir / "orig.png")
                ann_path  = str(tmp_dir / "annotated.png")
                
                orig_pil.save(orig_path)
                ann_pil.save(ann_path)
                
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 16)
                pdf.cell(0, 10, "EdgeVision - Aerospace Inspection Report", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 8, f"Image: {uploaded_file.name}", ln=True)
                pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                pdf.ln(4)
                
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Visual Inspection Mapping", ln=True)
                img_w = 85
                y0 = pdf.get_y()
                pdf.image(orig_path, x=10, y=y0, w=img_w)
                pdf.image(ann_path, x=10 + img_w + 10, y=y0, w=img_w)
                
                if heatmap_pil:
                    heat_path = str(tmp_dir / "heatmap.png")
                    heatmap_pil.save(heat_path)
                    pdf.ln(img_w * 0.75 + 5)
                    pdf.cell(0, 8, "EigenCAM Heatmap Overlay", ln=True)
                    pdf.image(heat_path, x=10, y=pdf.get_y(), w=img_w)
                    pdf.ln(img_w * 0.75 + 10)
                else:
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
                
                # Cleanup formatting for FPDF
                clean_text = re.sub(r'[*#_`]', '', openai_report)
                clean_text = clean_text.replace('- ', '  ')
                clean_text = re.sub(r'[^\x00-\x7F]+', ' ', clean_text)
                pdf.multi_cell(0, 6, clean_text)
                
                pdf_name = f"inspection_{int(time.time())}.pdf"
                pdf_full_path = REPORTS_DIR / pdf_name
                pdf.output(str(pdf_full_path))
                pdf_path = str(pdf_full_path)
                
            except Exception as e:
                st.warning(f"Failed to generate PDF Report: {e}")

            # Append to history log
            history = load_history()
            history.append({
                "timestamp":  datetime.datetime.now().isoformat(),
                "image":      uploaded_file.name,
                "detections": detections,
                "summary":    openai_report,
                "pdf_report": pdf_path,
            })
            save_history(history)

            # Store in session state
            st.session_state.analysis_results = {
                "image_name": uploaded_file.name,
                "detections": detections,
                "original_img": orig_pil,
                "annotated_img": ann_pil,
                "heatmap_img": heatmap_pil,
                "highest_risk": highest_risk,
                "total_defects": len(detections),
                "report": openai_report,
                "pdf_url": pdf_path
            }
            st.session_state.vision_report = vision_analysis
            st.session_state.acknowledged_escalation = False
            
            # Clean up temp file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"Error executing inspection pipeline: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            st.stop()

# ── Render Dashboard Tab ────────────────────────────────────────────────

with tab_dashboard:
    if st.session_state.analysis_results is None:
        st.markdown("""
            <div style="text-align: center; padding: 60px 20px; color: #9ca3af;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity: 0.5; margin-bottom: 16px;"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
                <h3>Awaiting aircraft component image.</h3>
                <p>Upload a photo of an aerospace component above to run real-time YOLO detection and GenAI maintenance planning.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        results = st.session_state.analysis_results
        
        # Display side-by-side images
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="glass-header">📸 Original Image</div>', unsafe_allow_html=True)
            if results["original_img"]:
                st.image(results["original_img"], use_container_width=True)
            else:
                st.info("Original image not loaded from cache.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="glass-header">⚡ Detection Map</div>', unsafe_allow_html=True)
            if results["annotated_img"]:
                st.image(results["annotated_img"], use_container_width=True)
            else:
                st.info("Detection mapping image unavailable.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Metrics and Detections Details
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
                <div class="metric-box">
                    <span style="color: #9ca3af; font-size: 0.9rem;">Total Detections</span>
                    <div class="metric-val">{results["total_defects"]}</div>
                </div>
            """, unsafe_allow_html=True)
        with col_m2:
            risk = results["highest_risk"]
            color = RISK_COLORS_HEX.get(risk, "#9ca3af")
            st.markdown(f"""
                <div class="metric-box">
                    <span style="color: #9ca3af; font-size: 0.9rem;">Highest Risk Level</span>
                    <div class="metric-val" style="color: {color};">{risk}</div>
                </div>
            """, unsafe_allow_html=True)

        # Defect List Details
        st.markdown('<div class="glass-card" style="margin-top: 24px;">', unsafe_allow_html=True)
        st.markdown('<div class="glass-header">📋 Detailed Findings</div>', unsafe_allow_html=True)
        
        requires_manual = False
        
        if results["detections"]:
            for d in results["detections"]:
                # Check anomaly/low conf escalation
                is_bg = "background" in d["class"].lower()
                is_low_conf = d["confidence"] < 0.40
                if is_bg or is_low_conf:
                    requires_manual = True
                
                risk_badge = f'<span class="badge badge-{d["risk"].lower()}">{d["risk"]}</span>'
                st.markdown(f"""
                    <div class="defect-item">
                        <div class="defect-info">
                            <span class="defect-title">{d["class"]}</span>
                            <span class="defect-meta">Confidence: {d["confidence"]*100:.1f}% &nbsp;&bull;&nbsp; Relative Area: {d["area_ratio"]*100:.2f}%</span>
                        </div>
                        {risk_badge}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#10b981; font-weight:500;">No critical structural defects found above confidence threshold.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Manual Review / Escalation Alert Banner
        if requires_manual:
            st.markdown("""
                <div class="alert-banner">
                    <div class="alert-header">⚠️ ACTION REQUIRED: MANUAL INSPECTION ESCALATION</div>
                    <div class="alert-text">
                        A low confidence detection (< 40%) or structural anomaly was flagged.
                        Airworthiness regulations require human review of this component before flight clearance.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Interactive button in Streamlit style
            col_esc, _ = st.columns([1, 3])
            with col_esc:
                if not st.session_state.acknowledged_escalation:
                    if st.button("🚨 Escalate & Acknowledge", use_container_width=True):
                        st.session_state.acknowledged_escalation = True
                        st.success("Escalation logged to inspection history!")
                        st.rerun()
                else:
                    st.success("✓ Escalated & Acknowledged")

        # Technical Report Display
        st.markdown('<div class="glass-card" style="margin-top: 24px;">', unsafe_allow_html=True)
        st.markdown('<div class="glass-header">📑 Technical MRO Guidance</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="report-section">
                <span style="font-family: 'Outfit', sans-serif; font-weight:600; color:#0ea5e9;">TECHNICAL RECONSTRUCTION & MAPPING</span>
                <div style="margin-top:12px;"></div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(results["report"])
        
        # Download PDF Button
        if results["pdf_url"] and os.path.exists(results["pdf_url"]):
            with open(results["pdf_url"], "rb") as f:
                pdf_data = f.read()
            st.download_button(
                label="📥 Download PDF Inspection Report",
                data=pdf_data,
                file_name=os.path.basename(results["pdf_url"]),
                mime="application/pdf",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Vision API native display (Accordion for Advanced Analysis)
        with st.expander("⚡ Advanced Vision AI Analysis (Raw Pixel Interpretation)"):
            if st.session_state.vision_report:
                st.markdown(st.session_state.vision_report)
            else:
                st.info("Vision report unavailable.")

# ── Render CAM Tab ──────────────────────────────────────────────────────

with tab_cam:
    if st.session_state.analysis_results is None:
        st.info("Upload an image first to visualize activation heatmaps.")
    else:
        results = st.session_state.analysis_results
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="glass-header">🔬 EigenCAM Explainability Heatmap</div>', unsafe_allow_html=True)
        st.markdown("""
            This heatmap displays activation maps from the deep feature layers of the model (Layer -2). 
            Warm regions (red/orange) indicate where the neural network focused its attention when analyzing structural integrity.
        """)
        if results["heatmap_img"]:
            st.image(results["heatmap_img"], use_container_width=True)
        else:
            st.warning("EigenCAM activation map was not generated. Check if YOLO-CAM cloned correctly.")
        st.markdown('</div>', unsafe_allow_html=True)

# ── Render Chat Tab ─────────────────────────────────────────────────────

with tab_chat:
    if not st.session_state.openai_configured:
        st.warning("Please configure your OpenAI API Key in the sidebar to enable chat functionality.")
    elif st.session_state.analysis_results is None:
        st.info("Chat will be enabled after you upload a component image and run the detection pipeline.")
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="glass-header">💬 Chat with EdgeVision MRO Assistant</div>', unsafe_allow_html=True)
        st.markdown("Ask questions regarding the maintenance recommendations, defect severity, or structural repair steps.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Display conversational chat history
        # Skip system and initial prompt seed lines for better presentation
        for msg in st.session_state.chat_history:
            if msg["role"] == "system":
                continue
            if "Inspection data:" in msg["content"] and msg["role"] == "user":
                continue
            
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input
        if user_query := st.chat_input("Ask about this defect or recommended repairs:"):
            # Display user query
            with st.chat_message("user"):
                st.markdown(user_query)

            # Request LLM response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("*Typing...*")
                
                # Append user query to state
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                client = load_openai_client(st.session_state.api_key)
                try:
                    response = client.chat.completions.create(
                        model=st.session_state.gpt_model,
                        messages=st.session_state.chat_history,
                        temperature=0.3
                    )
                    answer = response.choices[0].message.content
                    message_placeholder.markdown(answer)
                    
                    # Store assistant response in history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    message_placeholder.markdown(f"<span style='color:#ef4444;'>Chat Error: {e}</span>", unsafe_allow_html=True)
