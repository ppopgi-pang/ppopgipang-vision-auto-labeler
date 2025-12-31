import yaml
from pathlib import Path
from pipelines.base import PipelineStep
from modules.detector.yolo_world import YoloDetector
from modules.detector.bbox_utils import crop_image
from modules.storage.metadata_store import MetadataStore
from domain.image import ImageItem

class DetectPipeline(PipelineStep):
    def __init__(self, config_path: str = "configs/detector.yaml"):
        project_root = Path(__file__).resolve().parent.parent
        config_file = project_root / config_path
        
        if config_file.exists():
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            print(f"Warning: Config file {config_file} not found. Using defaults.")
            self.config = {}
            
        self.detector = YoloDetector(self.config)
        self.store = MetadataStore()
        
        self.save_crops = self.config.get("save_crops", True)
        self.crop_padding = self.config.get("crop_padding", 0)

    def run(self, images: list[ImageItem]) -> list[dict]:
        """
        Input: list of ImageItem
        Output: list of dicts with keys: {image_id, path, bboxes: [BoundingBox], crop_paths: [Path]}
        """
        print(f"--- DetectPipeline Start ({len(images)} images) ---")
        
        results = []
        
        for img_item in images:
            if not img_item.path:
                continue
                
            # Run detection
            bboxes = self.detector.detect(img_item)
            
            crop_paths = []
            if bboxes and self.save_crops:
                for idx, bbox in enumerate(bboxes):
                    # Define unique crop path: data/crops/{label}/{image_id}_{idx}.jpg
                    # Clean label for path
                    label_clean = "".join(c for c in bbox.label if c.isalnum() or c in (' ', '_', '-')).strip()
                    
                    crop_filename = f"{img_item.id}_{idx}.jpg"
                    crop_path = Path("data/crops") / label_clean / crop_filename
                    
                    success = crop_image(img_item.path, bbox, crop_path, padding=self.crop_padding)
                    if success:
                        crop_paths.append(str(crop_path))
            
            # Create result entry
            result_entry = {
                "image_id": img_item.id,
                "original_path": str(img_item.path),
                "bboxes": [
                    {
                        "label": b.label,
                        "confidence": b.confidence,
                        "xyxy": b.xyxy
                    } for b in bboxes
                ],
                "crop_paths": crop_paths
            }
            results.append(result_entry)
            
        # Save results
        output_path = Path("data/artifacts/bboxes.json")
        self.store.save(results, output_path)
            
        print(f"--- DetectPipeline Complete. Processed {len(images)} images. Results saved to {output_path} ---")
        return results
