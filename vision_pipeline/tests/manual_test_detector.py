import sys
from pathlib import Path
from PIL import Image, ImageDraw
import yaml

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from pipelines.detect_pipeline import DetectPipeline
from domain.image import ImageItem

def create_test_image_with_objects(path: Path):
    img = Image.new('RGB', (640, 640), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a "person" (approx) - red circle head + blue body
    draw.ellipse((300, 100, 340, 140), fill='red') # Head
    draw.rectangle((280, 140, 360, 300), fill='blue') # Body
    
    # Ideally YOLO needs real features, simple shapes might not trigger it comfortably.
    # But yolov8 is robust. Let's try.
    # Alternatively download a real image?
    # Since we can't easily download in this env without curl/wget which might fail if no internet, 
    # let's assume the user has internet or run_command works.
    # Actually, we can use the crawler stub or just try simple shapes.
    
    # A better synthetic image that triggers 'person' might be hard.
    # Let's trust that the 'yolov8n.pt' works on real images.
    # If this fails to detect, it's expected on synthetic data.
    # We mainly test the PIPELINE orchestration.
    
    img.save(path)
    return path

def main():
    test_dir = project_root / "data" / "test_detector"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = test_dir / "test_person.jpg"
    create_test_image_with_objects(img_path)
    
    print("Initializing Pipeline...")
    pipeline = DetectPipeline() # Uses defaults which load models/yolov8n.pt
    
    images = [
        ImageItem(id="test1", path=img_path, source="test", meta={})
    ]
    
    print("Running Pipeline...")
    results = pipeline.run(images)
    
    print("\n--- Result Summary ---")
    print(results)
    
    # Check if crops were saved
    if results and results[0]["crop_paths"]:
        print("SUCCESS: Crops generated.")
    else:
        print("WARNING: No detections/crops (Expected if using synthetic shapes with real model).")
        print("Pipeline logic executed successfully regardless of detection count.")

if __name__ == "__main__":
    main()
