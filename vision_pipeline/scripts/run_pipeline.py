import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from domain.job import Job
from services.pipeline_runner import PipelineRunner

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Vision Pipeline 실행")
    parser.add_argument("--keywords", type=str, nargs='+', required=True, help="검색 키워드 (예: 'miku figure' 'anime doll')")
    parser.add_argument("--target", type=str, default="object", help="CLIP/LLM용 대상 객체 이름 (예: 'miku')")
    parser.add_argument("--limit", type=int, default=10, help="크롤링할 이미지 수")

    args = parser.parse_args()

    # 첫 번째 키워드로 간단한 job ID 생성
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
