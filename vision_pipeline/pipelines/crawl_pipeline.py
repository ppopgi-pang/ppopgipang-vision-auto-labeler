from pipelines.base import PipelineStep
from domain.job import Job
from domain.image import ImageItem
from modules.crawler.google import GoogleCrawler
from modules.crawler.naver import NaverCrawler
from modules.storage.image_store import ImageStore
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from pathlib import Path
from config import settings

class CrawlPipeline(PipelineStep):
    def __init__(self):
        self.google_crawler = GoogleCrawler()
        self.naver_crawler = NaverCrawler()
        self.image_store = ImageStore()

    def _has_existing_data(self, raw_dir: str) -> bool:
        """Check if raw directory has any image files"""
        if not os.path.exists(raw_dir):
            return False

        for keyword_dir in os.listdir(raw_dir):
            keyword_path = os.path.join(raw_dir, keyword_dir)
            if os.path.isdir(keyword_path):
                # Check if directory has any image files
                files = [f for f in os.listdir(keyword_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                if files:
                    return True
        return False

    def _load_existing_images(self, raw_dir: str) -> list[ImageItem]:
        """Load existing images from raw directory"""
        images = []

        for keyword_dir in os.listdir(raw_dir):
            keyword_path = os.path.join(raw_dir, keyword_dir)
            if os.path.isdir(keyword_path):
                # Convert keyword_slug back to keyword (replace _ with space)
                keyword = keyword_dir.replace("_", " ")

                # Load all image files
                for filename in os.listdir(keyword_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        filepath = os.path.join(keyword_path, filename)
                        # Extract ID from filename (remove extension)
                        image_id = os.path.splitext(filename)[0]

                        item = ImageItem(
                            id=image_id,
                            path=Path(filepath),
                            keyword=keyword,
                            source="existing"  # Mark as existing data
                        )
                        images.append(item)

        print(f"[CrawlPipeline] 기존 이미지 {len(images)}개를 로드했습니다.")
        return images

    def run(self, job: Job) -> list[ImageItem]:
        print(f"CrawlPipeline 실행 중 - job: {job}")

        # Check if raw data already exists
        raw_dir = os.path.join(settings.output_dir, "raw")
        if self._has_existing_data(raw_dir):
            print(f"\n{'='*60}")
            print(f"[CrawlPipeline] 기존 크롤링 데이터가 발견되었습니다: {raw_dir}")
            print(f"[CrawlPipeline] 크롤링을 스킵하고 기존 데이터를 로드합니다...")
            print(f"{'='*60}\n")
            return self._load_existing_images(raw_dir)

        images: list[ImageItem] = []

        # job 또는 config에 기반하여 사용할 크롤러 결정
        # 현재는 두 크롤러 모두 사용하거나 Job의 플래그에 기반한다고 가정
        # Job 정의가 여기서 완전히 보이지 않으므로 키워드에 대해 두 크롤러 모두 검색하는 것을 기본으로 함

        if job.keywords:
            # 크롤러를 병렬로 실행
            crawlers = [
                ("Google", self.google_crawler),
                ("Naver", self.naver_crawler)
            ]

            # 세부 프로그레스바 (position=1: 전체 프로그레스바 아래, leave=False: 완료 후 제거)
            with tqdm(total=len(crawlers), desc="크롤러", unit="개", position=1, leave=False) as pbar:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {
                        executor.submit(crawler.fetch, job.keywords): name
                        for name, crawler in crawlers
                    }

                    for future in as_completed(futures):
                        crawler_name = futures[future]
                        try:
                            result = future.result()
                            images.extend(result)
                            pbar.set_postfix_str(f"{crawler_name}: {len(result)}개")
                            pbar.update(1)
                        except Exception as e:
                            pbar.set_postfix_str(f"{crawler_name}: 오류")
                            pbar.update(1)

        print(f"총 크롤링된 이미지: {len(images)}")

        # 이미지 저장
        self.image_store.save_raw(images)

        return images
