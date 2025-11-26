from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os

app = FastAPI(title="Grocery Detector API", version="1.0.0")

# CORS middleware to allow requests from Streamlit and other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (cached)
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

# Global variables for model
predictor = None
metadata = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor, metadata
    try:
        predictor, metadata = load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

@app.get("/")
async def root():
    return {
        "message": "Grocery Detector API", 
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, jpeg, png)")
    
    try:
        # Read and validate image
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run detection
        outputs = predictor(img_cv)
        instances = outputs["instances"].to("cpu")
        
        # Visualize results
        v = Visualizer(img_cv[:, :, ::-1], metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(instances)
        result_img = out.get_image()[:, :, ::-1]
        
        # Convert result image to base64
        _, buffer = cv2.imencode('.jpg', result_img)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Convert original image to base64
        original_img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        _, original_buffer = cv2.imencode('.jpg', original_img_rgb)
        original_image_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
        # Prepare detection results
        detections = []
        if len(instances) > 0:
            for i in range(len(instances)):
                bbox = instances.pred_boxes.tensor.cpu().numpy()[i].astype(int)
                detection = {
                    "item": metadata.thing_classes[int(instances.pred_classes[i])],
                    "confidence": float(instances.scores[i]),
                    "bbox": bbox.tolist(),
                    "bbox_formatted": f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                }
                detections.append(detection)
        
        return JSONResponse(content={
            "detections_count": len(instances),
            "detections": detections,
            "original_image": original_image_base64,
            "result_image": result_image_base64,
            "message": f"Found {len(instances)} item(s)" if len(instances) > 0 else "No items detected above 70% confidence"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)