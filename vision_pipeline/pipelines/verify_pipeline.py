import yaml
import json
from pathlib import Path
from pipelines.base import PipelineStep
from modules.llm.verifier import LLMVerifier
from modules.storage.metadata_store import MetadataStore
from domain.label import LabelResult

class VerifyPipeline(PipelineStep):
    def __init__(self, config_path: str = "configs/llm.yaml"):
        project_root = Path(__file__).resolve().parent.parent
        config_file = project_root / config_path
        
        if config_file.exists():
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            print(f"Warning: Config file {config_file} not found. Using defaults.")
            self.config = {}
            
        self.verifier = LLMVerifier(self.config)
        self.store = MetadataStore()

    def run(self, detection_results: list[dict]) -> list[LabelResult]:
        """
        Input: Output of DetectPipeline (list of dicts)
        Output: list of LabelResult
        """
        print(f"--- VerifyPipeline Start ({len(detection_results)} items) ---")
        
        final_results = []
        
        for item in detection_results:
            image_id = item.get("image_id")
            crop_paths = item.get("crop_paths", [])
            bboxes = item.get("bboxes", [])
            
            # We assume bboxes and crop_paths are aligned by index
            # Or we can just iterate crop_paths. 
            # But we need the label from bbox.
            
            if len(crop_paths) != len(bboxes):
                print(f"[VerifyPipeline] Warning: Mismatch crops/bboxes for {image_id}. Skipping.")
                continue
                
            for i, crop_path in enumerate(crop_paths):
                bbox_label = bboxes[i]["label"]
                
                # Run Verification
                result = self.verifier.verify_image(crop_path, label=bbox_label)
                result.image_id = image_id # Attach parent ID
                
                if result.verified:
                    print(f"[VerifyPipeline] Verified {image_id} as {bbox_label} (Conf: {result.confidence})")
                else:
                    # print(f"[VerifyPipeline] Rejected {image_id} as {bbox_label}: {result.reason}")
                    pass
                    
                final_results.append(result)
                
        # Save results
        output_path = Path("data/artifacts/verification.json")
        self.store.save([r.to_dict() for r in final_results], output_path)
            
        print(f"--- VerifyPipeline Complete. Verified {len([r for r in final_results if r.verified])}/{len(final_results)} items. ---")
        return final_results
