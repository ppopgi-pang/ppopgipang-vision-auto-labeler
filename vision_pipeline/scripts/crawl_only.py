import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from vision_pipeline.domain.job import Job
from vision_pipeline.pipelines.crawl_pipeline import CrawlPipeline

if __name__ == "__main__":
    job = Job(
        keywords=["Hatsune Miku doll"],
        target="miku",
    )
    
    pipeline = CrawlPipeline()
    pipeline.run(job)
