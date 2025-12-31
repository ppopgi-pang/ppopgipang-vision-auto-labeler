from pipelines.base import PipelineStep
from domain.job import Job
from domain.image import ImageItem
from modules.crawler.google import GoogleCrawler
from modules.crawler.naver import NaverCrawler
from modules.storage.image_store import ImageStore

class CrawlPipeline(PipelineStep):
    def __init__(self):
        self.google_crawler = GoogleCrawler()
        self.naver_crawler = NaverCrawler()
        self.image_store = ImageStore()

    def run(self, job: Job) -> list[ImageItem]:
        print(f"Running CrawlPipeline for job: {job}")
        
        images: list[ImageItem] = []
        
        # Determine which crawler to use based on job or config
        # For now, let's assume we use both or based on some flag in Job
        # Since Job definition isn't fully visible here, I'll default to searching both for the keywords
        
        if job.keywords:
            print("Fetching from Google...")
            images.extend(self.google_crawler.fetch(job.keywords))
            
            print("Fetching from Naver...")
            images.extend(self.naver_crawler.fetch(job.keywords))
            
        print(f"Total images crawled: {len(images)}")
        
        # Save images
        self.image_store.save_raw(images)
        
        return images
