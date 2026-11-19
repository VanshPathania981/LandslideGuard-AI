import io
import zipfile
import tempfile
import json
from pathlib import Path

import streamlit as st
import torch
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from report_generator import generate_prediction_pdf
from model import UNet
from prediction_history import initialize_history, save_prediction, load_history
from prediction_history import (
    initialize_history,
    save_prediction,
    load_history,
    HISTORY_FILE
)
# =========================================================
# 1. STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LandslideGuard AI",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PATHS AND SETTINGS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "models" / "best_unet.pth"
DATA_DIR = BASE_DIR / "data" / "predictions"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = 0.70


# =========================================================
# 3. MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = UNet()

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    channel_mean = checkpoint.get("channel_mean") if isinstance(checkpoint, dict) else None
    channel_std = checkpoint.get("channel_std") if isinstance(checkpoint, dict) else None

    if channel_mean is not None:
        channel_mean = torch.as_tensor(
            channel_mean, dtype=torch.float32
        ).reshape(1, 14, 1, 1).to(DEVICE)

    if channel_std is not None:
        channel_std = torch.as_tensor(
            channel_std, dtype=torch.float32
        ).reshape(1, 14, 1, 1).to(DEVICE)

    return model, channel_mean, channel_std, checkpoint


# =========================================================
# 4. IMAGE AND METRIC HELPERS
# =========================================================

def prepare_rgb(image):
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3 or image.shape[-1] != 14:
        raise ValueError("Expected an image with 14 channels.")

    # RGB display uses channels 3, 2, 1.
    rgb = image[:, :, [3, 2, 1]].copy()
    output = np.zeros_like(rgb, dtype=np.uint8)

    for channel in range(3):
        band = rgb[:, :, channel]
        valid = np.isfinite(band)

        if not np.any(valid):
            continue

        low, high = np.percentile(band[valid], [2, 98])

        if high <= low:
            high = low + 1e-6

        scaled = np.clip((band - low) / (high - low), 0, 1)
        scaled[~valid] = 0
        output[:, :, channel] = (scaled * 255).astype(np.uint8)

    return output


def calculate_coverage(mask):
    mask = np.asarray(mask)

    if mask.dtype != bool:
        mask = mask > 0

    predicted_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    coverage = predicted_pixels / total_pixels * 100 if total_pixels else 0

    return predicted_pixels, total_pixels, coverage


def calculate_metrics(predicted, ground_truth):
    predicted = np.asarray(predicted, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)

    tp = int(np.logical_and(predicted, ground_truth).sum())
    fp = int(np.logical_and(predicted, ~ground_truth).sum())
    fn = int(np.logical_and(~predicted, ground_truth).sum())
    tn = int(np.logical_and(~predicted, ~ground_truth).sum())

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }


def classify_coverage(coverage):
    # Illustrative image-coverage category, not an official warning level.
    if coverage < 2:
        return "LOW"
    elif coverage < 5:
        return "MODERATE"
    elif coverage < 10:
        return "HIGH"
    return "VERY HIGH"


# =========================================================
# 5. CACHED PREDICTION DATA
# =========================================================

@st.cache_data(show_spinner=False)
def load_probability(sample_id):
    path = DATA_DIR / f"{sample_id}_probability.npy"

    if not path.exists():
        raise FileNotFoundError(path.name)

    probability = np.load(path)

    if probability.shape != (128, 128):
        raise ValueError(f"Invalid probability shape: {probability.shape}")

    return probability.astype(np.float32)


@st.cache_data(show_spinner=False)
def load_ground_truth(sample_id):
    path = DATA_DIR / f"{sample_id}_groundtruth.png"

    if not path.exists():
        raise FileNotFoundError(path.name)

    return np.array(Image.open(path).convert("L")) > 127


@st.cache_data(show_spinner=False)
def load_rgb(sample_id):
    path = DATA_DIR / f"{sample_id}_rgb.png"

    if not path.exists():
        raise FileNotFoundError(path.name)

    return np.array(Image.open(path).convert("RGB"))


@st.cache_data(show_spinner=False)
def calculate_threshold_metrics(sample_ids):
    rows = []
    thresholds = np.round(np.arange(0.10, 0.951, 0.05), 2)

    probabilities = []
    ground_truths = []

    for sample_id in sample_ids:
        probabilities.append(load_probability(sample_id).reshape(-1))
        ground_truths.append(load_ground_truth(sample_id).reshape(-1))

    if not probabilities:
        return pd.DataFrame()

    probabilities = np.concatenate(probabilities)
    ground_truths = np.concatenate(ground_truths)

    for threshold in thresholds:
        predicted = probabilities >= threshold
        metrics = calculate_metrics(predicted, ground_truths)

        rows.append({
            "Threshold": float(threshold),
            "Dice": metrics["Dice"],
            "IoU": metrics["IoU"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "Accuracy": metrics["Accuracy"]
        })

    return pd.DataFrame(rows)


# =========================================================
# 6. CHECK DATA AND LOAD MODEL
# =========================================================

if not DATA_DIR.exists():
    st.error(f"Prediction data folder not found: {DATA_DIR}")
    st.info("Create the folder and place your exported prediction files inside it.")
    st.stop()

sample_ids = sorted(
    path.name[:-len("_rgb.png")]
    for path in DATA_DIR.glob("*_rgb.png")
)

if not sample_ids:
    st.error("No validation prediction samples found.")
    st.info("Expected files such as sample_1_rgb.png in data/predictions.")
    st.stop()

try:
    model, channel_mean, channel_std, checkpoint = load_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# =========================================================
# 7. GLOBAL PROFESSIONAL UI DESIGN
# =========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080F1D 0%, #0B1425 55%, #101C30 100%);
    color: #E2E8F0;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101B2E 0%, #0B1424 100%);
    border-right: 1px solid #263B57;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #94A3B8;
}

section[data-testid="stSidebar"] .stRadio > label {
    color: #94A3B8;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 7px;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 10px 12px;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: #172A42;
    border-color: #29445F;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
    color: #CBD5E1;
    font-size: 13px;
    font-weight: 550;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
    background: #15304A;
    border: 1px solid #087EA8;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) p {
    color: #38BDF8;
    font-weight: 700;
}

/* Sidebar brand and information */
.sidebar-brand {
    padding: 15px 12px;
    background: linear-gradient(135deg, #172A42, #102034);
    border: 1px solid #29445F;
    border-radius: 13px;
    margin-bottom: 20px;
}

.sidebar-brand-title {
    font-size: 20px;
    font-weight: 800;
    color: #F1F5F9;
    letter-spacing: -0.5px;
}

.sidebar-brand-subtitle {
    color: #94A3B8;
    font-size: 10px;
    letter-spacing: 1.4px;
    margin-top: 4px;
}

.sidebar-info {
    background: #111F32;
    border: 1px solid #263B57;
    border-radius: 11px;
    padding: 13px;
    margin-top: 8px;
}

.sidebar-info-title {
    color: #E2E8F0;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 10px;
}

.sidebar-info-row {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid #24344D;
    font-size: 11px;
}

.sidebar-info-label {
    color: #94A3B8;
}

.sidebar-info-value {
    color: #CBD5E1;
    font-weight: 600;
    text-align: right;
}

.sidebar-status {
    background: #12352F;
    border: 1px solid #166534;
    color: #4ADE80;
    border-radius: 7px;
    padding: 8px 10px;
    font-size: 11px;
    font-weight: 700;
    margin-top: 12px;
}

/* Typography */
h1 {
    color: #F1F5F9 !important;
    font-size: 2.1rem !important;
    font-weight: 750 !important;
    letter-spacing: -0.8px;
}

h2 {
    color: #E2E8F0 !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}

h3 {
    color: #CBD5E1 !important;
    font-size: 1.15rem !important;
    font-weight: 650 !important;
}

p, li {
    color: #CBD5E1;
    line-height: 1.7;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #F1F5F9;
    letter-spacing: -1px;
}

.subtitle {
    color: #94A3B8;
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 5px;
}

.section-label {
    color: #38BDF8;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* Metrics and panels */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17243A, #111C2F);
    border: 1px solid #263B57;
    padding: 19px 17px;
    border-radius: 13px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.18);
    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: #38BDF8;
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(56, 189, 248, 0.09);
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 12px;
}

div[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
    font-size: 26px !important;
    font-weight: 750;
}

.risk-panel {
    background: linear-gradient(145deg, #17243A, #111C2F);
    border: 1px solid #263B57;
    padding: 20px;
    border-radius: 13px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #263B57 !important;
    border-radius: 13px !important;
}

div[data-testid="stExpander"] {
    background: #111C2F;
    border: 1px solid #263B57;
    border-radius: 11px;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #0284C7, #0369A1);
    color: #FFFFFF;
    border: 1px solid #0EA5E9;
    border-radius: 9px;
    padding: 9px 18px;
    font-weight: 650;
    transition: all 0.2s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0EA5E9, #0284C7);
    color: #FFFFFF;
    border-color: #7DD3FC;
    transform: translateY(-2px);
}

/* Inputs and uploader */
.stTextInput input, .stNumberInput input,
.stSelectbox div[data-baseweb="select"], .stTextArea textarea {
    background-color: #111C2F !important;
    color: #E2E8F0 !important;
    border-radius: 9px !important;
    border-color: #334761 !important;
}

div[data-testid="stFileUploader"] {
    background: #111C2F;
    border: 1px dashed #38BDF8;
    border-radius: 13px;
    padding: 13px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #263B57;
    border-radius: 12px;
    overflow: hidden;
}

.status {
    display: inline-block;
    background: #12352F;
    color: #4ADE80;
    padding: 9px 15px;
    border: 1px solid #166534;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
}

hr {
    border-color: #263B57;
    margin-top: 1.2rem;
    margin-bottom: 1.2rem;
}

div[data-testid="stImage"] img {
    border-radius: 11px;
    border: 1px solid #263B57;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
    border: 1px solid #334761;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1 {
        font-size: 1.65rem !important;
    }

    .main-title {
        font-size: 26px;
    }

    div[data-testid="stMetric"] {
        padding: 13px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 8. PROFESSIONAL SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-title">⛰️ LandslideGuard</div>
    <div class="sidebar-brand-subtitle">
        AI-POWERED LANDSLIDE ANALYSIS
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    '<div class="section-label">MAIN MENU</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Prediction Explorer",
        "Model Performance",
        "Live Prediction",
        "Prediction History",
        "About & Methodology"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

best_epoch_html = ""

if isinstance(checkpoint, dict) and "epoch" in checkpoint:
    best_epoch_html = f"""
    <div class="sidebar-info-row">
        <span class="sidebar-info-label">Best Epoch</span>
        <span class="sidebar-info-value">{checkpoint['epoch']}</span>
    </div>
    """


st.sidebar.markdown("---")
st.sidebar.markdown("### MODEL CONFIGURATION")

st.sidebar.markdown("**Architecture:** U-Net")
st.sidebar.markdown("**Input Channels:** 14")
st.sidebar.markdown("**Resolution:** 128 × 128")
st.sidebar.markdown(f"**Compute Device:** {DEVICE}")

# Display best epoch if available
if best_epoch_html:
    st.sidebar.caption("Best epoch information available in model training results.")

st.sidebar.success("● MODEL LOADED")

st.sidebar.divider()
st.sidebar.caption("LANDSLIDEGUARD AI · ACADEMIC PROJECT")



# =========================================================
# 9. MAIN HEADER
# =========================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        '<div class="main-title">⛰️ LandslideGuard AI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">INTELLIGENT LANDSLIDE MONITORING & SEGMENTATION PLATFORM</div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div class="status">● SYSTEM READY</div>',
        unsafe_allow_html=True
    )

st.divider()


# =========================================================
# 10. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.markdown("""
        <div class="main-title">Landslide Monitoring Overview</div>
        <div class="subtitle">
            SATELLITE INTELLIGENCE · U-NET SEGMENTATION · VALIDATION ANALYSIS
        </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("""
        <div class="status">
            ● MODEL READY — VALIDATION PREDICTIONS AVAILABLE
        </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Validation Samples",
        f"{len(sample_ids):,}",
        help="Number of available validation prediction samples."
    )
    col2.metric(
        "Input Channels", "14",
        help="Number of satellite input channels."
    )
    col3.metric(
        "Image Resolution", "128 × 128",
        help="Spatial dimensions of each input image."
    )
    col4.metric(
        "Default Threshold",
        f"{DEFAULT_THRESHOLD:.2f}",
        help="Default probability threshold for binary segmentation."
    )

    st.write("")
    st.divider()
    st.subheader("Satellite Prediction Explorer")
    st.caption(
        "Select a validation sample and adjust the segmentation "
        "threshold to explore model predictions."
    )

    control_col1, control_col2 = st.columns([1, 2])

    with control_col1:
        selected_sample = st.selectbox(
            "Select Validation Sample",
            sample_ids,
            key="overview_sample"
        )

    with control_col2:
        threshold = st.slider(
            "Prediction Confidence Threshold",
            min_value=0.10,
            max_value=0.95,
            value=DEFAULT_THRESHOLD,
            step=0.05,
            key="overview_threshold"
        )

    try:
        rgb = load_rgb(selected_sample)
        probability = load_probability(selected_sample)
        ground_truth = load_ground_truth(selected_sample)
        predicted_mask = probability >= threshold

        metrics = calculate_metrics(predicted_mask, ground_truth)
        predicted_pixels, total_pixels, coverage = calculate_coverage(predicted_mask)

        st.markdown("### Prediction Visualization")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Satellite RGB Image**")
            st.image(rgb, use_container_width=True)

        with col2:
            st.markdown("**Predicted Landslide Mask**")
            st.image(
                predicted_mask.astype(np.uint8) * 255,
                use_container_width=True
            )

        st.write("")
        st.divider()
        st.markdown("### Sample Analytics")

        a, b, c = st.columns(3)
        a.metric("Predicted Pixels", f"{predicted_pixels:,}")
        b.metric("Predicted Coverage", f"{coverage:.2f}%")
        c.metric("Dice Score", f"{metrics['Dice']:.4f}")

        st.markdown("### Predicted Pixel Coverage")
        st.progress(min(max(coverage / 100, 0.0), 1.0))
        st.caption(
            f"{predicted_pixels:,} of {total_pixels:,} pixels "
            f"({coverage:.2f}%) classified as landslide."
        )
        st.info(
            "Coverage represents the percentage of image pixels classified "
            "as landslide. It is not a geographic risk estimate or an "
            "official landslide warning."
        )

    except Exception as error:
        st.error(f"Could not load sample: {error}")


# =========================================================
# 11. PREDICTION EXPLORER PAGE
# =========================================================

elif page == "Prediction Explorer":

    st.header("Prediction Explorer")
    st.write(
        "Inspect satellite images, predicted segmentation masks, "
        "probability maps and ground-truth masks."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_sample = st.selectbox(
            "Choose sample", sample_ids, key="explorer_sample"
        )

    with col2:
        threshold = st.slider(
            "Segmentation Threshold",
            min_value=0.10,
            max_value=0.95,
            value=DEFAULT_THRESHOLD,
            step=0.05,
            key="explorer_threshold"
        )

    try:
        rgb = load_rgb(selected_sample)
        probability = load_probability(selected_sample)
        ground_truth = load_ground_truth(selected_sample)
        predicted_mask = probability >= threshold

        metrics = calculate_metrics(predicted_mask, ground_truth)
        predicted_pixels, total_pixels, coverage = calculate_coverage(predicted_mask)
        risk_level = classify_coverage(coverage)

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dice Score", f"{metrics['Dice']:.4f}")
        col2.metric("IoU Score", f"{metrics['IoU']:.4f}")
        col3.metric("Precision", f"{metrics['Precision']:.4f}")
        col4.metric("Recall", f"{metrics['Recall']:.4f}")

        st.subheader("Image Analysis")
        c1, c2 = st.columns(2)

        with c1:
            st.image(rgb, caption="Satellite RGB", use_container_width=True)

        with c2:
            st.image(
                predicted_mask.astype(np.uint8) * 255,
                caption="Predicted Mask",
                use_container_width=True
            )

        c3, c4 = st.columns(2)

        with c3:
            st.image(
                ground_truth.astype(np.uint8) * 255,
                caption="Ground Truth",
                use_container_width=True
            )

        with c4:
            st.image(
                (probability * 255).clip(0, 255).astype(np.uint8),
                caption="Probability Map",
                use_container_width=True
            )

        st.divider()
        st.subheader("Prediction Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Landslide Pixels", f"{predicted_pixels:,}")
        c2.metric("Total Pixels", f"{total_pixels:,}")
        c3.metric("Image Coverage", f"{coverage:.2f}%")

        st.markdown(
            f"""
            <div class="risk-panel">
                <h3>Image Coverage Category: {risk_level}</h3>
                <p>
                    This category is based only on the percentage of pixels
                    classified as landslide in this image. It is not an
                    official warning or a real-world hazard classification.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()
        st.subheader("Download Prediction Mask")

        mask_buffer = io.BytesIO()
        Image.fromarray(predicted_mask.astype(np.uint8) * 255).save(
            mask_buffer, format="PNG"
        )

        st.download_button(
            "Download Predicted Mask",
            data=mask_buffer.getvalue(),
            file_name=f"{selected_sample}_predicted_mask.png",
            mime="image/png"
        )

    except Exception as error:
        st.error(f"Prediction Explorer error: {error}")


# =========================================================
# 12. MODEL PERFORMANCE PAGE
# =========================================================

elif page == "Model Performance":

    st.header("Model Performance")
    st.write(
        "Validation metrics are calculated by comparing the exported "
        "predictions against their ground-truth masks."
    )

    threshold = st.slider(
        "Evaluation Threshold",
        min_value=0.10,
        max_value=0.95,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        key="performance_threshold"
    )

    try:
        with st.spinner("Calculating validation metrics..."):
            results_df = calculate_threshold_metrics(tuple(sample_ids))

            if results_df.empty:
                st.warning("No validation metrics are available.")
                st.stop()

            all_probabilities = []
            all_ground_truths = []

            for sample_id in sample_ids:
                all_probabilities.append(
                    load_probability(sample_id).reshape(-1)
                )
                all_ground_truths.append(
                    load_ground_truth(sample_id).reshape(-1)
                )

            all_probabilities = np.concatenate(all_probabilities)
            all_ground_truths = np.concatenate(all_ground_truths)
            all_predictions = all_probabilities >= threshold

            metrics = calculate_metrics(all_predictions, all_ground_truths)

        st.subheader("Validation Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Dice", f"{metrics['Dice']:.4f}")
        c2.metric("IoU", f"{metrics['IoU']:.4f}")
        c3.metric("Precision", f"{metrics['Precision']:.4f}")
        c4.metric("Recall", f"{metrics['Recall']:.4f}")
        c5.metric("Accuracy", f"{metrics['Accuracy']:.4f}")

        st.caption(
            f"Metrics calculated at threshold {threshold:.2f} "
            f"across {len(sample_ids)} validation samples."
        )

        st.divider()
        st.subheader("Confusion Matrix")

        confusion = np.array([
            [metrics["TN"], metrics["FP"]],
            [metrics["FN"], metrics["TP"]]
        ])

        fig, ax = plt.subplots(figsize=(6, 4))
        image_plot = ax.imshow(confusion)
        ax.set_title("Pixel-Level Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Landslide", "Landslide"])
        ax.set_yticklabels(["Non-Landslide", "Landslide"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{confusion[i, j]:,}",
                    ha="center", va="center", color="white"
                )

        fig.colorbar(image_plot, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("Threshold Performance Analysis")

        chart_metrics = results_df.set_index("Threshold")[
            ["Dice", "IoU", "Precision", "Recall"]
        ]
        st.line_chart(chart_metrics)

        best_row = results_df.loc[results_df["Dice"].idxmax()]
        st.success(
            f"Best validation Dice threshold: {best_row['Threshold']:.2f} "
            f"(Dice: {best_row['Dice']:.4f})"
        )

        st.dataframe(
            results_df.style.format({
                "Threshold": "{:.2f}",
                "Dice": "{:.4f}",
                "IoU": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "Accuracy": "{:.4f}"
            }),
            use_container_width=True
        )

        st.info(
            "The threshold comparison uses the validation dataset. "
            "The best validation threshold is not an independent test result."
        )

    except Exception as error:
        st.error(f"Model Performance error: {error}")


# =========================================================
# 13. LIVE PREDICTION PAGE
# =========================================================

elif page == "Live Prediction":

    st.header("Live Landslide Prediction")
    st.write(
        "Upload a satellite HDF5 image containing 14 input channels "
        "to generate a segmentation prediction."
    )

    st.info(
        "Required input: HDF5 (.h5) file with a dataset named 'img' "
        "and shape (128, 128, 14). PNG and JPG files are not accepted."
    )

    threshold = st.slider(
        "Prediction Threshold",
        min_value=0.10,
        max_value=0.95,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        key="live_threshold"
    )

    uploaded_file = st.file_uploader(
        "Upload 14-channel HDF5 satellite image",
        type=["h5", "hdf5"]
    )

    if uploaded_file is not None:
        st.write(f"Selected file: **{uploaded_file.name}**")

        if st.button("Run Landslide Prediction", type="primary"):
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".h5"
                ) as temporary_file:
                    temporary_file.write(uploaded_file.getvalue())
                    temp_path = temporary_file.name

                if not h5py.is_hdf5(temp_path):
                    raise ValueError(
                        "Invalid file format! Please upload a valid "
                        "14-channel HDF5 satellite image."
                    )

                with h5py.File(temp_path, "r") as h5_file:
                    if "img" not in h5_file:
                        raise ValueError(
                            "The HDF5 file does not contain the required 'img' dataset."
                        )
                    satellite_image = np.array(h5_file["img"])

                if satellite_image.shape != (128, 128, 14):
                    raise ValueError(
                        f"Invalid image shape: {satellite_image.shape}. "
                        "Expected (128, 128, 14)."
                    )

                if not np.isfinite(satellite_image).all():
                    raise ValueError(
                        "Input image contains NaN or infinite values."
                    )

                input_tensor = torch.from_numpy(
                    satellite_image.astype(np.float32)
                ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

                if channel_mean is not None and channel_std is not None:
                    safe_std = torch.where(
                        channel_std == 0,
                        torch.ones_like(channel_std),
                        channel_std
                    )
                    input_tensor = (input_tensor - channel_mean) / safe_std

                with torch.no_grad():
                    output = model(input_tensor)

                    if output.ndim != 4 or output.shape[1] != 1:
                        raise ValueError(
                            f"Unexpected model output shape: {tuple(output.shape)}"
                        )

                    probability = (
                        torch.sigmoid(output).squeeze().cpu().numpy()
                    )

                if probability.shape != (128, 128):
                    raise ValueError(
                        f"Unexpected probability map shape: {probability.shape}"
                    )

                if not np.isfinite(probability).all():
                    raise ValueError("Model produced invalid probability values.")

                predicted_mask = probability >= threshold
                predicted_pixels, total_pixels, coverage = calculate_coverage(
                    predicted_mask
                )
                save_prediction(
                                    image_name=uploaded_file.name,
                                    threshold=threshold,
                                    total_pixels=total_pixels,
                                    predicted_pixels=predicted_pixels,
                                    coverage=coverage
                                )
                
                risk_level = classify_coverage(coverage)
                pdf_report = generate_prediction_pdf(
                image_name=uploaded_file.name,
                threshold=threshold,
                total_pixels=total_pixels,
                predicted_pixels=predicted_pixels,
                coverage=coverage
            )

                st.download_button(
                    label="Download Prediction Report (PDF)",
                    data=pdf_report,
                    file_name="landslide_prediction_report.pdf",
                    mime="application/pdf"
                )
                rgb_image = prepare_rgb(satellite_image)

                binary_image = Image.fromarray(
                    predicted_mask.astype(np.uint8) * 255
                )
                probability_image = Image.fromarray(
                    (probability * 255).clip(0, 255).astype(np.uint8)
                )

                overlay = rgb_image.copy()
                red_layer = np.zeros_like(overlay)
                red_layer[:, :, 0] = 255
                overlay[predicted_mask] = (
                    0.5 * overlay[predicted_mask]
                    + 0.5 * red_layer[predicted_mask]
                ).astype(np.uint8)
                overlay_image = Image.fromarray(overlay)

                st.success("Prediction completed successfully!")
                st.divider()
                st.subheader("Prediction Statistics")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Predicted Pixels", f"{predicted_pixels:,}")
                c2.metric("Total Pixels", f"{total_pixels:,}")
                c3.metric("Predicted Coverage", f"{coverage:.2f}%")
                c4.metric("Threshold", f"{threshold:.2f}")

                st.markdown(
                    f"""
                    <div class="risk-panel">
                        <h3>Image Coverage Category: {risk_level}</h3>
                        <p>
                            This is an image-level pixel coverage category,
                            not an official landslide warning or geographic
                            hazard assessment.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.divider()
                st.subheader("Prediction Visualizations")

                c1, c2 = st.columns(2)
                with c1:
                    st.image(
                        rgb_image,
                        caption="Satellite RGB Image",
                        use_container_width=True
                    )
                with c2:
                    st.image(
                        binary_image,
                        caption="Predicted Binary Mask",
                        use_container_width=True
                    )

                c3, c4 = st.columns(2)
                with c3:
                    st.image(
                        probability_image,
                        caption="Probability Map",
                        use_container_width=True
                    )
                with c4:
                    st.image(
                        overlay_image,
                        caption="Landslide Overlay",
                        use_container_width=True
                    )

                st.divider()

                mask_buffer = io.BytesIO()
                binary_image.save(mask_buffer, format="PNG")

                st.download_button(
                    label="Download Prediction Mask",
                    data=mask_buffer.getvalue(),
                    file_name="landslide_prediction_mask.png",
                    mime="image/png"
                )

                # Complete ZIP report
                report_buffer = io.BytesIO()

                with zipfile.ZipFile(
                    report_buffer, "w", zipfile.ZIP_DEFLATED
                ) as report_zip:

                    rgb_buffer = io.BytesIO()
                    Image.fromarray(rgb_image).save(rgb_buffer, format="PNG")
                    report_zip.writestr(
                        "satellite_rgb.png", rgb_buffer.getvalue()
                    )

                    mask_report_buffer = io.BytesIO()
                    binary_image.save(mask_report_buffer, format="PNG")
                    report_zip.writestr(
                        "predicted_mask.png", mask_report_buffer.getvalue()
                    )

                    probability_buffer = io.BytesIO()
                    probability_image.save(probability_buffer, format="PNG")
                    report_zip.writestr(
                        "probability_map.png", probability_buffer.getvalue()
                    )

                    overlay_buffer = io.BytesIO()
                    overlay_image.save(overlay_buffer, format="PNG")
                    report_zip.writestr(
                        "landslide_overlay.png", overlay_buffer.getvalue()
                    )

                    summary = f"""
LANDSLIDEGUARD AI - PREDICTION REPORT

Input file: {uploaded_file.name}
Model: U-Net
Input channels: 14
Image resolution: 128 x 128
Device: {DEVICE}

Prediction threshold: {threshold:.2f}
Predicted landslide pixels: {predicted_pixels}
Total pixels: {total_pixels}
Predicted image coverage: {coverage:.2f}%

Image coverage category: {risk_level}

IMPORTANT:
This is an AI-generated image-level segmentation result.
It is not an official landslide warning.
It does not provide geographic coordinates or a real-world
hazard classification.
"""
                    report_zip.writestr("prediction_summary.txt", summary)

                st.download_button(
                    label="📦 Download Complete Prediction Report",
                    data=report_buffer.getvalue(),
                    file_name="LandslideGuard_Prediction_Report.zip",
                    mime="application/zip"
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

            finally:
                if temp_path is not None:
                    try:
                        Path(temp_path).unlink(missing_ok=True)
                    except OSError:
                        pass



elif page == "Prediction History":

    st.title("Prediction History")
    st.caption("Review and analyze previously generated predictions.")

    history = load_history()

    if not history:
        st.info("No prediction records found. Run a prediction first.")

    else:
        import pandas as pd

        history_df = pd.DataFrame(history)

        # Search predictions
        search_text = st.text_input(
            "Search by image name",
            placeholder="Enter image filename..."
        )

        if search_text:
            history_df = history_df[
                history_df["image_name"].str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ]

        if not history_df.empty:

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Predictions",
                len(history_df)
            )

            col2.metric(
                "Average Coverage",
                f"{history_df['coverage'].mean():.2f}%"
            )

            col3.metric(
                "Average Threshold",
                f"{history_df['threshold'].mean():.2f}"
            )

            st.subheader("Prediction Records")

            display_df = history_df[
                [
                    "date",
                    "image_name",
                    "threshold",
                    "predicted_pixels",
                    "coverage"
                ]
            ].sort_values("date", ascending=False)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

             
            st.subheader("Prediction Details")

            record_options = {
                f"{row['image_name']} | {row['date']}": row["id"]
                for _, row in history_df.iterrows()
            }

            if record_options:
                selected_record = st.selectbox(
                    "Select a prediction",
                    options=list(record_options.keys())
                )

                selected_id = record_options[selected_record]

                selected_data = history_df[
                    history_df["id"] == selected_id
                ].iloc[0]

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Predicted Pixels",
                    f"{selected_data['predicted_pixels']:,}"
                )

                col2.metric(
                    "Coverage",
                    f"{selected_data['coverage']:.2f}%"
                )

                col3.metric(
                    "Threshold",
                    f"{selected_data['threshold']:.2f}"
                )
   

            st.divider()

            if st.button("Clear All Prediction History", type="secondary"):

                with open(HISTORY_FILE, "w") as file:
                    json.dump([], file, indent=4)

                st.success("Prediction history cleared successfully.")

                st.rerun()

            # Coverage comparison
            st.subheader("Predicted Coverage Comparison")

            chart_df = display_df[
                ["image_name", "coverage"]
            ].set_index("image_name")

            st.bar_chart(chart_df)

# ==========================================
# PREDICTION TREND ANALYSIS
# ==========================================

            st.divider()

            st.subheader("Prediction Trend Analysis")

            trend_df = history_df.copy()

            trend_df["date"] = pd.to_datetime(
                trend_df["date"],
                errors="coerce"
            )

            trend_df = trend_df.dropna(subset=["date"])

            trend_df = trend_df.sort_values("date")

            if not trend_df.empty:

                trend_chart = trend_df[
                    ["date", "coverage"]
                ].set_index("date")

                st.line_chart(
                    trend_chart,
                    use_container_width=True
                )

                st.caption(
                    "This chart shows predicted pixel coverage "
                    "across recorded image predictions."
                )

            else:
                st.info("Not enough valid date records to display a trend.")

            # Download history
            csv_data = history_df.to_csv(index=False)

            st.download_button(
                "Download Prediction History (CSV)",
                data=csv_data,
                file_name="prediction_history.csv",
                mime="text/csv"
            )

        else:
            st.warning("No matching prediction records found.")

# =========================================================
# 14. ABOUT & METHODOLOGY PAGE
# =========================================================

elif page == "About & Methodology":

    st.header("About LandslideGuard AI")

    st.write(
        """
        LandslideGuard AI is an AI-powered landslide segmentation
        prototype that uses a U-Net deep learning model to identify
        potential landslide pixels in satellite imagery.
        """
    )

    st.divider()
    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Architecture", "U-Net")
    col2.metric("Input Channels", "14")
    col3.metric("Image Resolution", "128 × 128")

    st.write(
        """
        The system processes multi-channel satellite imagery and
        generates a pixel-level probability map. A configurable
        threshold converts that probability map into a binary
        landslide segmentation mask.
        """
    )

    st.divider()
    st.subheader("Model Architecture")

    st.markdown("""
    **U-Net Encoder–Decoder Architecture**

    - Input layer: 14 channels
    - Encoder: 32, 64, 128 and 256 feature channels
    - Bottleneck: 512 feature channels
    - Decoder: Transposed convolutions and skip connections
    - Output: One-channel segmentation logits
    - Activation: Sigmoid during prediction
    """)

    st.divider()
    st.subheader("Training Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", "3,039")
    col2.metric("Validation Samples", "760")
    col3.metric("Training Epochs", "10")

    st.write("Best checkpoint: Epoch 7")
    st.write(
        """
        The dataset was divided into a custom training and validation
        split. The validation set was used to evaluate model performance
        and select the prediction threshold.
        """
    )

    st.warning(
        "The reported validation results are not independent test-set "
        "results. The project does not claim external benchmark validation."
    )

    st.divider()
    st.subheader("Validation Performance")

    performance_data = pd.DataFrame({
        "Metric": [
            "Dice Score",
            "Landslide IoU",
            "Precision",
            "Recall",
            "Pixel Accuracy"
        ],
        "Score": [0.7263, 0.5703, 0.7080, 0.7456, 0.9867]
    })

    st.dataframe(
        performance_data.style.format({"Score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Reference validation metrics at threshold 0.70. "
        "Values may vary slightly depending on the exported predictions."
    )

    st.divider()
    st.subheader("System Workflow")

    workflow = [
        "Upload 14-channel satellite HDF5 image",
        "Validate image format and dimensions",
        "Normalize input using training statistics",
        "Process image through trained U-Net",
        "Generate pixel-level probability map",
        "Apply configurable prediction threshold",
        "Generate binary landslide segmentation mask",
        "Display visualizations and export prediction report"
    ]

    for index, step in enumerate(workflow, start=1):
        st.markdown(f"**{index}.** {step}")

    st.divider()
    st.subheader("System Limitations")

    st.markdown("""
    - The current model processes images at 128 × 128 resolution.
    - Predictions depend on input image quality and training data.
    - Input files must contain the expected 14-channel image structure.
    - The current dataset files do not provide geographic coordinates
      or georeferencing information.
    - Image coverage categories are illustrative, not official risk levels.
    - The application does not provide real-time satellite connectivity.
    - Predictions must not be treated as official landslide warnings.
    """)

    st.divider()
    st.subheader("Future Scope")

    st.markdown("""
    - Integration with georeferenced satellite imagery.
    - Real-time satellite data ingestion.
    - GIS-based landslide visualization.
    - Historical landslide monitoring.
    - Advanced model evaluation on independent datasets.
    - Deployment as a web-based monitoring platform.
    """)

    st.divider()
    st.success("LandslideGuard AI | Academic Deep Learning Project")

    st.divider()
    st.markdown(
        """
        <div style="text-align:center; color:#94A3B8; padding:15px;">
            <h4 style="color:#38BDF8;">LANDSLIDEGUARD AI</h4>
            <p>AI-Powered Landslide Segmentation & Risk Monitoring Prototype</p>
            <p>Developed as an Academic Deep Learning Project</p>
            <p style="font-size:12px;">
                This system provides experimental image-level predictions.
                It is not an official landslide warning or disaster-management system.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
