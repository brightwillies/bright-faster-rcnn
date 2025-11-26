# app.py
import streamlit as st
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
import cv2
import numpy as np
from PIL import Image
import torch
import os

# -------------------------------
# CONFIG (same as your training)
# -------------------------------
@st.cache_resource
def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = "model_final.pth"        # ← your trained weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3          # cheerios, soup, candles
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set class names
    metadata = MetadataCatalog.get("grocery_val")
    metadata.thing_classes = ["cheerios", "soup", "candles"]

    predictor = DefaultPredictor(cfg)
    return predictor, metadata

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Grocery Items Classifieer", layout="centered")
st.title("Grocery Items Detector")
st.markdown("**Trained only on synthetic data → 82.6% mAP on real photos**")
st.markdown("Upload any image – works on phones, shelves, tables!")

predictor, metadata = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("Detecting..."):
        outputs = predictor(img_cv)
        instances = outputs["instances"].to("cpu")

    # Visualize
    from detectron2.utils.visualizer import Visualizer, ColorMode
    v = Visualizer(img_cv[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(instances)
    result_img = out.get_image()[:, :, ::-1]

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(result_img, caption=f"{len(instances)} detection(s)", use_column_width=True)

    # Print results
    if len(instances) > 0:
        st.success(f"Found {len(instances)} grocery item(s):")
        for i in range(len(instances)):
            cls = instances.pred_classes[i].item()
            score = instances.scores[i].item()
            box = instances.pred_boxes.tensor[i].cpu().numpy().astype(int)
            st.write(f"• **{metadata.thing_classes[cls]}** – {score:.1%} confidence")
    else:
        st.info("No cheerios, soup, or candles detected above 70% confidence.")