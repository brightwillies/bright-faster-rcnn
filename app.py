from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import subprocess
import sys

app = FastAPI(title="Grocery Detector API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
predictor = None
metadata = None

def install_detectron2_colab_style():
    """Install Detectron2 using the exact Colab method that worked"""
    try:
        # First check if it's already installed
        import detectron2
        print("✅ Detectron2 is already installed")
        return True
    except ImportError:
        print("🚀 Installing Detectron2 using Colab method...")
        try:
            # This is the EXACT command that worked in Colab
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", 
                "git+https://github.com/facebookresearch/detectron2.git"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Detectron2 installed successfully!")
                return True
            else:
                print(f"❌ Detectron2 installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error installing Detectron2: {e}")
            return False

def load_model():
    """Load Detectron2 model - same as Colab"""
    try:
        # Install Detectron2 first if needed
        if not install_detectron2_colab_style():
            raise Exception("Failed to install Detectron2")
        
        # Now import Detectron2 components
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.utils.visualizer import Visualizer, ColorMode
        
        print("📦 Loading model configuration...")
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = "model_final.pth"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.DEVICE = "cpu"  # Render uses CPU
        
        # Create metadata
        metadata = type('', (), {})()
        metadata.thing_classes = ["cheerios", "soup", "candles"]
        
        print("🔨 Creating predictor...")
        predictor = DefaultPredictor(cfg)
        
        print("✅ Model loaded successfully!")
        return predictor, metadata
        
    except Exception as e:
        print(f"❌ Error in load_model: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor, metadata
    try:
        print("=" * 60)
        print("🚀 Starting Grocery Detector API")
        print("=" * 60)
        
        print(f"Python: {sys.version}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        
        # Check for model file
        if not os.path.exists("model_final.pth"):
            raise FileNotFoundError("model_final.pth not found!")
        print("📁 Model file: FOUND")
        
        # Load model (this will install Detectron2 if needed)
        predictor, metadata = load_model()
        print("🎉 Startup completed - API is ready!")
        
    except Exception as e:
        print(f"💥 CRITICAL: Startup failed - {e}")
        print("❌ API will start but detection endpoints will fail")

@app.get("/")
async def root():
    return {
        "message": "Grocery Detector API", 
        "status": "running",
        "model_loaded": predictor is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if predictor is not None else "model_not_loaded",
        "model_loaded": predictor is not None
    }

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded - check server logs")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Upload an image file")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run detection
        outputs = predictor(img_cv)
        instances = outputs["instances"].to("cpu")
        
        # Visualize
        from detectron2.utils.visualizer import Visualizer, ColorMode
        v = Visualizer(img_cv[:, :, ::-1], metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(instances)
        result_img = out.get_image()[:, :, ::-1]
        
        # Convert to base64
        _, result_buffer = cv2.imencode('.jpg', result_img)
        result_b64 = base64.b64encode(result_buffer).decode()
        
        _, orig_buffer = cv2.imencode('.jpg', cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        orig_b64 = base64.b64encode(orig_buffer).decode()
        
        # Prepare results
        detections = []
        if len(instances) > 0:
            for i in range(len(instances)):
                bbox = instances.pred_boxes.tensor.cpu().numpy()[i].astype(int)
                detections.append({
                    "item": metadata.thing_classes[int(instances.pred_classes[i])],
                    "confidence": float(instances.scores[i]),
                    "bbox": bbox.tolist(),
                    "bbox_formatted": f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                })
        
        return {
            "detections_count": len(instances),
            "detections": detections,
            "original_image": orig_b64,
            "result_image": result_b64,
            "message": f"Found {len(instances)} item(s)" if len(instances) > 0 else "No items detected"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)