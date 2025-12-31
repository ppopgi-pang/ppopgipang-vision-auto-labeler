import sys
from pathlib import Path
from PIL import Image
import yaml
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from pipelines.filter_pipeline import FilterPipeline
from domain.image import ImageItem

def create_red_square(path):
    img = Image.new('RGB', (300, 300), color='red')
    img.save(path)
    return path

def main():
    test_dir = project_root / "data" / "test_images_classifier"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create a "Red Square" image
    img_path = create_red_square(test_dir / "red_square.png")
    
    # 2. Define ImageItem with keyword "red" (Should pass)
    img_good = ImageItem(
        id="good",
        path=img_path,
        source="test",
        keyword="red color",
        meta={}
    )
    
    # 3. Define ImageItem with keyword "blue" (Should fail)
    img_bad = ImageItem(
        id="bad",
        path=img_path,
        source="test",
        keyword="blue color", # Trying to match "red square" to "blue color" -> low score expected?
        # Actually CLIP is good at color. "a photo of blue color" vs "red square" -> low.
        meta={}
    )

    print("Initializing Pipeline with CLIP...")
    
    # We use a lower threshold to ensure 'good' passes even if synthetic image is weird
    # But for 'red color' prompt and red image, it should be very high.
    config_data = {
        "dedup": {"method": "phash", "hash_size": 8, "threshold": 5},
        "quality": {"min_width": 100, "min_height": 100, "min_file_size_kb": 0, "max_aspect_ratio": 3.0},
        "classifier": {
            "model_name": "openai/clip-vit-base-patch32", 
            "threshold": 0.2, # Very low threshold for testing
            "device": "cpu"
        }
    }
    
    config_path = project_root / "configs" / "test_classifier_pipeline.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    pipeline = FilterPipeline(config_path="configs/test_classifier_pipeline.yaml")
    
    print("Running Pipeline...")
    # NOTE: DEDUPLICATOR will remove the second image because it's a duplicate of the first!
    # We must skip dedup or use different images.
    # Let's bypass dedup by commenting it out in pipeline or making images different.
    # To make them different phash, let's just make another image.
    
    img_path_2 = test_dir / "red_square_2.png"
    Image.new('RGB', (300, 300), color=(255, 0, 1)).save(img_path_2) # slightly different red? No, identical
    # Let's just use the same image item list but know that dedup comes first.
    # Wait, if I want to test Classifier, I should probably invoke Classifier directly OR 
    # configure Dedup to be very loose or ensuring images are distinct.
    
    # Let's invoke Classifier directly for precise testing.
    from modules.filter.classifier import Classifier
    classifier = Classifier(config_data["classifier"])
    
    results = classifier.keep_positive([img_good, img_bad])
    
    print("\n--- Result Summary ---")
    ids = [img.id for img in results]
    print(f"Kept IDs: {ids}")
    
    # Expected: "good" should be kept (red image, keyword 'red color')
    # "bad" should be rejected (red image, keyword 'blue color') - Score for "a photo of blue color" should be lower than "not blue color" or just low.
    
    if "good" in ids:
        print("SUCCESS: Good image kept.")
    else:
        print("FAILURE: Good image rejected.")
        
    if "bad" not in ids:
        print("SUCCESS: Bad image rejected.")
    else:
        print("FAILURE: Bad image kept (maybe threshold too low or CLIP confused).")

if __name__ == "__main__":
    main()
