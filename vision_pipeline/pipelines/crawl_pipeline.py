from pipelines.base import PipelineStep
from domain.job import Job
from domain.image import ImageItem
from modules.crawler.google import GoogleCrawler
from modules.crawler.naver import NaverCrawler
from modules.storage.image_store import ImageStore
from concurrent.futures import ThreadPoolExecutor, as_completed

class CrawlPipeline(PipelineStep):
    def __init__(self):
        self.google_crawler = GoogleCrawler()
        self.naver_crawler = NaverCrawler()
        self.image_store = ImageStore()

    def run(self, job: Job) -> list[ImageItem]:
        print(f"CrawlPipeline 실행 중 - job: {job}")

        images: list[ImageItem] = []

        # job 또는 config에 기반하여 사용할 크롤러 결정
        # 현재는 두 크롤러 모두 사용하거나 Job의 플래그에 기반한다고 가정
        # Job 정의가 여기서 완전히 보이지 않으므로 키워드에 대해 두 크롤러 모두 검색하는 것을 기본으로 함

        if job.keywords:
            print("Google과 Naver에서 병렬로 가져오는 중...")

            # 크롤러를 병렬로 실행
            crawlers = [
                ("Google", self.google_crawler),
                ("Naver", self.naver_crawler)
            ]

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
                        print(f"{crawler_name} 크롤링 완료: {len(result)}개 이미지")
                    except Exception as e:
                        print(f"{crawler_name} 크롤링 오류: {e}")

        print(f"총 크롤링된 이미지: {len(images)}")

        # 이미지 저장
        self.image_store.save_raw(images)

        return images
