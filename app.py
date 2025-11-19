import streamlit as st
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
import cv2
import numpy as np
from PIL import Image
import os

# Load model (cached)
@st.cache_resource
def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = "model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Metadata
    metadata = model_zoo.get_coco_metadata()
    metadata.thing_classes = ["cheerios", "soup", "candles"]
    
    predictor = DefaultPredictor(cfg)
    return predictor, metadata

st.set_page_config(page_title="Grocery Detector", layout="wide")
st.title("🛒 Grocery Items Detector")
st.markdown("**Trained on synthetic data only — 82.6% mAP on real photos**")

predictor, metadata = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Detecting..."):
        outputs = predictor(img_cv)
        instances = outputs["instances"].to("cpu")
    
    # Visualize
    v = Visualizer(img_cv[:, :, ::-1], metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(instances)
    result_img = out.get_image()[:, :, ::-1]
    
    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    with col2:
        st.image(result_img, caption=f"{len(instances)} detection(s)", use_column_width=True)
    
    # Table
    if len(instances) > 0:
        st.success(f"**Found {len(instances)} grocery item(s):**")
        df = pd.DataFrame({
            "Item": [metadata.thing_classes[int(c)] for c in instances.pred_classes],
            "Confidence": [f"{s:.1%}" for s in instances.scores],
            "Box": [f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]" for x1, y1, x2, y2 in instances.pred_boxes.tensor.cpu().numpy().astype(int)]
        })
        st.table(df)
    else:
        st.info("No items detected above 70% confidence.")