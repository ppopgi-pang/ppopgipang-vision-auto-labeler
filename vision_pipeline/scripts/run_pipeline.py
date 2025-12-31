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
    parser.add_argument("--keywords", type=str, nargs='+', required=True, help="Search keywords (e.g., 'miku figure' 'anime doll')")
    parser.add_argument("--target", type=str, default="object", help="Target object name for CLIP/LLM (e.g., 'miku')")
    parser.add_argument("--limit", type=int, default=10, help="Number of images to crawl")
    
    args = parser.parse_args()
    
    # Create a simple job ID from the first keyword
    first_kw = args.keywords[0].replace(' ', '_')
    
    job = Job(
        keywords=args.keywords,
        target_class=args.target,
        limit=args.limit,
        job_id=f"job_{args.target}_{first_kw}"
    )
    
    runner = PipelineRunner()
    runner.run(job)

if __name__ == "__main__":
    main()
