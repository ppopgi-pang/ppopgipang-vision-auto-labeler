import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from domain.job import Job
from services.pipeline_runner import PipelineRunner

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run Vision Pipeline")
    parser.add_argument("--keyword", type=str, required=True, help="Search keyword (e.g., 'miku figure')")
    parser.add_argument("--target", type=str, default="object", help="Target object name for CLIP/LLM (e.g., 'miku')")
    parser.add_argument("--limit", type=int, default=10, help="Number of images to crawl")
    
    args = parser.parse_args()
    
    job = Job(
        keywords=[args.keyword],
        target_class=args.target,
        limit=args.limit,
        job_id=f"job_{args.target}_{args.keyword.replace(' ', '_')}"
    )
    
    runner = PipelineRunner()
    runner.run(job)

if __name__ == "__main__":
    main()
