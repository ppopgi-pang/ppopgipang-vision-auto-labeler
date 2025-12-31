from PIL import Image
from domain.bbox import BoundingBox
from pathlib import Path

def crop_image(image_path: str | Path, bbox: BoundingBox, output_path: str | Path, padding: int = 0):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            x1 = max(0, bbox.x1 - padding)
            y1 = max(0, bbox.y1 - padding)
            x2 = min(width, bbox.x2 + padding)
            y2 = min(height, bbox.y2 + padding)
            
            crop = img.crop((x1, y1, x2, y2))
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(output_path)
            
            return True
            
    except Exception as e:
        print(f"[BBoxUtils] Error cropping {image_path}: {e}")
        return False
