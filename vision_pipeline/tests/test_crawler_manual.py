import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from vision_pipeline.modules.crawler.google import GoogleCrawler
from vision_pipeline.modules.crawler.naver import NaverCrawler

def test_google():
    print("Testing Google Crawler...")
    crawler = GoogleCrawler()
    results = crawler.fetch(["cute cat"])
    print(f"Found {len(results)} images from Google.")
    for img in results[:3]:
        print(f" - {img.url}")

def test_naver():
    print("\nTesting Naver Crawler...")
    crawler = NaverCrawler()
    results = crawler.fetch(["korean landscape"])
    print(f"Found {len(results)} images from Naver.")
    for img in results[:3]:
        print(f" - {img.url}")

if __name__ == "__main__":
    test_google()
    test_naver()
