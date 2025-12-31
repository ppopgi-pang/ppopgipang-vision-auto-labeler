import sys
import os
from pathlib import Path

# Add project root to path to ensure imports work
sys.path.append(os.getcwd())

# ImageItem patch removed as domain model is updated
try:
    import vision_pipeline.domain.image
except ImportError:
    print("Could not import vision_pipeline.domain.image.")
    sys.exit(1)

try:
    from vision_pipeline.modules.crawler.google import GoogleCrawler
    from vision_pipeline.modules.crawler.naver import NaverCrawler
    from vision_pipeline.modules.storage.image_store import ImageStore
    from vision_pipeline.config import settings
except ImportError as e:
    print(f"Failed to import crawlers: {e}")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parents[1]
settings.output_dir = str(BASE_DIR / "vision_pipeline" / "data")
image_store = ImageStore()

def test_google():
    print("\n--- Testing Google Crawler ---")
    crawler = GoogleCrawler()
    try:
        results = crawler.fetch(["cute cat"])
        print(f"Found {len(results)} images from Google.")
        if results:
            print(f"Sample: {results[0]}")
            image_store.save_raw(results)
    except Exception as e:
        print(f"Google Crawler failed: {e}")

def test_naver():
    print("\n--- Testing Naver Crawler ---")
    crawler = NaverCrawler()
    try:
        results = crawler.fetch(["cute cat"])
        print(f"Found {len(results)} images from Naver.")
        if results:
            print(f"Sample: {results[0]}")
            image_store.save_raw(results)
    except Exception as e:
        print(f"Naver Crawler failed: {e}")

if __name__ == "__main__":
    test_google()
    test_naver()
