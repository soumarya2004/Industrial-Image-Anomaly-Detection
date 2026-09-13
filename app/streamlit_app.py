"""Streamlit demo app for the Industrial Image Anomaly Detection System.

Run locally with:
    streamlit run app/streamlit_app.py

Lets the user upload an image, pick a model (supervised classifier or
autoencoder anomaly detector), run inference, and see the prediction,
score, and localization heatmap side by side with the original (and, for
the autoencoder, the reconstruction).
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import streamlit as st
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import build_transforms
from src.data.preprocessing import ImageValidationError
from src.evaluation.experiment_log import load_all_experiments
from src.inference import load_autoencoder, load_classifier
from src.utils import get_device
from src.visualization.anomaly_map import generate_anomaly_visualization
from src.visualization.gradcam import generate_gradcam_overlay

st.set_page_config(page_title="Industrial Anomaly Detection", layout="wide")
CLASSIFIER_CKPT = "models/classifier/best_model.pt"
AUTOENCODER_CKPT = "models/autoencoder/best_model.pt"
@st.cache_resource
def _load_model(model_type: str):
    device = get_device(prefer_cuda=True)
    if model_type=="classifier":
        model, cfg=load_classifier(CLASSIFIER_CKPT, device)
    else:
        model, cfg=load_autoencoder(AUTOENCODER_CKPT, device)
    return model, cfg, device

def _default_threshold_from_logs()->float|None:
    #Pull the last logged autoencoder experiment's selected threshold, if any
    try:
        records=load_all_experiments()
        ae_records=[r for r in records if "autoencoder" in r["experiment_name"]]
        if ae_records:
            return ae_records[-1]["metrics"].get("selected_threshold")
    except Exception:
        pass
    return None

def main()->None:
    st.title("🔍 Industrial Image Anomaly Detection System")
    st.caption(
        "Upload a product image to check for defects using either a supervised "
        "CNN classifier or an unsupervised autoencoder anomaly detector."
    )
    with st.sidebar:
        st.header("Settings")
        model_choice=st.radio(
            "Model", options=["Supervised Classifier", "Autoencoder (Anomaly Detection)"]
        )
        model_type="classifier" if model_choice.startswith("Supervised") else "autoencoder"
        threshold=None
        if model_type=="autoencoder":
            default_threshold=_default_threshold_from_logs() or 0.01
            threshold=st.slider(
                "Anomaly threshold (reconstruction MSE)",
                min_value=0.0, max_value=max(0.1, default_threshold * 3),
                value=float(default_threshold), step=0.001, format="%.4f",
                help="Images with reconstruction error above this are flagged DEFECTIVE. "
                     "Defaults to the value selected on validation data during training.",
            )
        st.markdown("---")
        st.markdown(
            "**About:** compares two fundamentally different approaches to industrial "
            "visual inspection — a supervised classifier trained on labeled defects, and "
            "an unsupervised autoencoder trained only on normal product images."
        )
    uploaded_file = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_file is None:
        st.info("Upload an image to run inspection.")
        return
    try:
        model, cfg, device = _load_model(model_type)
    except FileNotFoundError:
        st.error(
            f"No trained checkpoint found for '{model_type}'. Train the model first:\n\n"
            f"`python -m src.training.train_{'classifier' if model_type == 'classifier' else 'autoencoder'} "
            f"--config configs/{'classifier' if model_type == 'classifier' else 'autoencoder'}.yaml`"
        )
        return
    image=Image.open(uploaded_file).convert("RGB")
    col_original, col_result=st.columns(2)
    with col_original:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
    if model_type=="classifier":
        transform=build_transforms(cfg["data"]["image_size"], train=False, normalize=True)
        image_tensor=transform(image).unsqueeze(0)
        with torch.no_grad():
            logits=model(image_tensor.to(device))
            probs=torch.softmax(logits, dim=1)[0]
        predicted_class=int(torch.argmax(probs).item())
        confidence=float(probs[predicted_class].item())
        label="DEFECTIVE" if predicted_class == 1 else "NORMAL"
        raw_transform=build_transforms(cfg["data"]["image_size"], train=False, normalize=False)
        image_rgb01=raw_transform(image).permute(1, 2, 0).numpy()
        overlay=generate_gradcam_overlay(
            model, image_tensor, image_rgb01, target_class=predicted_class, device=device
        )
        with col_result:
            st.subheader("Result")
            if label=="DEFECTIVE":
                st.error(f"Prediction: **{label}**")
            else:
                st.success(f"Prediction: **{label}**")
            st.metric("Confidence", f"{confidence * 100:.1f}%")
            st.metric("Defect probability", f"{float(probs[1].item()):.3f}")
        st.subheader("Grad-CAM: Where the model is looking")
        st.image(overlay, use_container_width=True, caption="Red/warm regions influenced the prediction most")
    else:  # autoencoder
        transform=build_transforms(cfg["data"]["image_size"], train=False, normalize=False)
        image_tensor=transform(image).unsqueeze(0)
        viz=generate_anomaly_visualization(model, image_tensor, device)
        is_anomalous=viz["anomaly_score"] >= threshold
        label="DEFECTIVE" if is_anomalous else "NORMAL"
        with col_result:
            st.subheader("Result")
            if label=="DEFECTIVE":
                st.error(f"Prediction: **{label}**")
            else:
                st.success(f"Prediction: **{label}**")
            st.metric("Anomaly score (reconstruction MSE)", f"{viz['anomaly_score']:.5f}")
            st.metric("Threshold", f"{threshold:.5f}")
        st.subheader("Reconstruction & Anomaly Localization")
        col_a, col_b, col_c=st.columns(3)
        with col_a:
            st.image(viz["original_rgb"], caption="Original (resized)", use_container_width=True)
        with col_b:
            st.image(viz["reconstructed_rgb"], caption="Autoencoder reconstruction", use_container_width=True)
        with col_c:
            overlay_rgb=viz["overlay_bgr"][:, :, ::-1]  # BGR -> RGB for st.image
            st.image(overlay_rgb, caption="Anomaly heatmap overlay", use_container_width=True)

if __name__ == "__main__":
    main()
