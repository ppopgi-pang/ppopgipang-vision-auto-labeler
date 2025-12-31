from ultralytics import YOLO
from pathlib import Path
from PIL import Image
from domain.image import ImageItem
from domain.bbox import BoundingBox

class YoloDetector:
    def __init__(self, config: dict):
        self.config = config
        self.model_path = config.get("model_path", "yolov8n.pt")
        self.conf_threshold = config.get("conf_threshold", 0.5)
        self.classes = config.get("classes", None) # List of class indices to filter, None = all
        
        print(f"[YoloDetector] Loading model from {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            print("[YoloDetector] Model loaded successfully.")
        except Exception as e:
            print(f"[YoloDetector] Failed to load model: {e}")
            raise e

    def detect(self, image_item: ImageItem) -> list[BoundingBox]:
        if not image_item.path:
            return []
            
        try:
            # Run inference
            results = self.model.predict(
                source=str(image_item.path), 
                conf=self.conf_threshold, 
                classes=self.classes,
                verbose=False
            )
            
            bboxes = []
            for result in results:
                for box in result.boxes:
                    # box.xyxy is [x1, y1, x2, y2]
                    coords = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    label = self.model.names[cls_id]
                    
                    bboxes.append(BoundingBox(
                        x1=coords[0],
                        y1=coords[1],
                        x2=coords[2],
                        y2=coords[3],
                        confidence=conf,
                        label=label
                    ))
            
            return bboxes
            
        except Exception as e:
            print(f"[YoloDetector] Error detecting in {image_item.path}: {e}")
            return []
