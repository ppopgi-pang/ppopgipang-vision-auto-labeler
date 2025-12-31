import yaml
from pathlib import Path
from domain.job import Job
from pipelines.crawl_pipeline import CrawlPipeline
from pipelines.filter_pipeline import FilterPipeline
from pipelines.detect_pipeline import DetectPipeline
from pipelines.verify_pipeline import VerifyPipeline

class PipelineRunner:
    def __init__(self, config_path: str = "configs/pipeline.yaml"):
        project_root = Path(__file__).resolve().parent.parent
        config_file = project_root / config_path
        
        if config_file.exists():
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            print(f"Warning: Config file {config_file} not found. Using defaults.")
            self.config = {}
            
        self.stage_config = self.config.get("stages", {})
        
        # Initialize Pipelines
        # (Lazy load could be better, but simple init is fine for now)
        self.crawl_pipeline = CrawlPipeline()
        self.filter_pipeline = FilterPipeline()
        self.detect_pipeline = DetectPipeline()
        self.verify_pipeline = VerifyPipeline()

    def run(self, job: Job):
        print(f"[{job.job_id}] Starting Pipeline Runner for {job.keywords}...")
        
        # 1. Crawl
        if self.stage_config.get("crawl", True):
            print("\n>>> Stage 1: Crawl")
            images = self.crawl_pipeline.run(job)
        else:
            print("\n>>> Stage 1: Crawl (Skipped)")
            # Should load from previous step if skipping?
            # For now, if skipped, we assume we can't proceed unless we load existing data.
            # But let's assume we usually run all or nothing for this MVP.
            images = []

        if not images and self.stage_config.get("crawl", True):
            print("No images found/crawled. Stopping.")
            return

        # 2. Filter
        if self.stage_config.get("filter", True):
            print("\n>>> Stage 2: Filter")
            # Ensure classifier checks against target class
            # We might need to inject target class into classifier config dynamically?
            # Or just update the images.keyword with job.target_class if missing
            for img in images:
                if not img.keyword: 
                    img.keyword = job.target_class
            
            filtered_images = self.filter_pipeline.run(images)
        else:
             print("\n>>> Stage 2: Filter (Skipped)")
             filtered_images = images # Pass through if skipped

        if not filtered_images:
            print("No images left after filtering. Stopping.")
            return

        # 3. Detect
        if self.stage_config.get("detect", True):
            print("\n>>> Stage 3: Detect")
            detection_results = self.detect_pipeline.run(filtered_images)
        else:
            print("\n>>> Stage 3: Detect (Skipped)")
            detection_results = []
            
        if not detection_results:
             print("No detections found. Stopping.")
             return

        # 4. Verify
        if self.stage_config.get("verify", True):
             print("\n>>> Stage 4: Verify")
             verification_results = self.verify_pipeline.run(detection_results)
        else:
             print("\n>>> Stage 4: Verify (Skipped)")
             verification_results = []
             
        print(f"\n[{job.job_id}] Pipeline Complete.")
        
        # Final Summary
        verified_count = len([r for r in verification_results if r.verified])
        print(f"Final Yield: {verified_count} verified labels from {len(images)} initial images.")
