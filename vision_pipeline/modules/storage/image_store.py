from domain.image import ImageItem

import requests
import uuid
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import settings
import glob

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

    def load_raw(self) -> list[ImageItem]:
        """raw 디렉토리에서 기존 이미지를 로드"""
        raw_dir = os.path.join(settings.output_dir, "raw")

        if not os.path.exists(raw_dir):
            print("[ImageStore] No raw directory found")
            return []

        images: list[ImageItem] = []

        # raw 디렉토리 내의 모든 하위 디렉토리(키워드별) 탐색
        for keyword_dir in os.listdir(raw_dir):
            keyword_path = os.path.join(raw_dir, keyword_dir)

            if not os.path.isdir(keyword_path):
                continue

            # 키워드 복원 (언더스코어를 공백으로)
            keyword = keyword_dir.replace("_", " ")

            # 디렉토리 내 모든 이미지 파일 찾기
            image_files = glob.glob(os.path.join(keyword_path, "*.*"))

            for image_file in image_files:
                # 파일명에서 ID 추출 (확장자 제거)
                file_id = os.path.splitext(os.path.basename(image_file))[0]

                item = ImageItem(
                    id=file_id,
                    keyword=keyword,
                    url="",  # 로컬 파일이므로 URL 없음
                    path=Path(image_file)
                )
                images.append(item)

        print(f"[ImageStore] Loaded {len(images)} images from raw directory")
        return images
