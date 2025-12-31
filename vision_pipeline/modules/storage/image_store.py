from domain.image import ImageItem

import requests
import uuid
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import settings

class ImageStore:
    def _download_one(self, item: ImageItem, index: int, total: int) -> bool:
        try:
            # 디렉토리 생성: settings.output_dir/raw/{keyword}
            keyword_slug = item.keyword.replace(" ", "_") if item.keyword else "unknown"
            save_dir = os.path.join(settings.output_dir, "raw", keyword_slug)
            os.makedirs(save_dir, exist_ok=True)

            # 파일명 생성
            # ID가 설정되지 않은 경우 생성
            if not item.id:
                item.id = str(uuid.uuid4())

            filename = f"{item.id}.jpg"  # 현재는 jpg로 가정, 또는 헤더에서 감지?
            # 더 나은 접근 방식은 content-type을 확인하거나 그대로 저장하고 나중에 감지하는 것.
            # 대부분이 이미지이므로 단순화를 위해 .jpg 사용.

            filepath = os.path.join(save_dir, filename)

            response = requests.get(item.url, timeout=10)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)

                item.path = Path(filepath)
                print(f"[ImageStore] Downloaded {index}/{total}: {item.url}")
                return True

            print(f"[ImageStore] Failed {index}/{total}: {item.url} ({response.status_code})")
            return False
        except Exception as e:
            print(f"[ImageStore] Error {index}/{total}: {item.url} ({e})")
            return False

    def save_raw(self, images: list[ImageItem]):
        print(f"[ImageStore] Saving {len(images)} raw images")
        download_items = [item for item in images if not item.path and item.url]
        total = len(download_items)

        if total == 0:
            print("[ImageStore] No new images to download.")
            return

        max_workers = max(1, int(getattr(settings, "max_workers", 4)))
        print(f"[ImageStore] Downloading {total} images with {max_workers} workers...")

        saved_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._download_one, item, idx, total)
                for idx, item in enumerate(download_items, start=1)
            ]
            for future in as_completed(futures):
                if future.result():
                    saved_count += 1

        print(f"[ImageStore] Saved {saved_count}/{total} raw images")
