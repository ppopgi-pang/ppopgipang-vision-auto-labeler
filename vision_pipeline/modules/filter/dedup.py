import imagehash
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from modules.filter.base import FilterStep
from domain.image import ImageItem
from config import settings


def _compute_hash_worker(img_path: str, hash_size: int) -> tuple[imagehash.ImageHash | None, str | None]:
    """
    ProcessPoolExecutor를 위한 모듈 레벨 함수 (picklable)
    """
    if not img_path or not Path(img_path).exists():
        return None, f"[Deduplicator] specific path not found: {img_path}"

    try:
        with Image.open(img_path) as pil_img:
            current_hash = imagehash.phash(pil_img, hash_size=hash_size)
        return current_hash, None
    except Exception as e:
        return None, f"[Deduplicator] Error processing {img_path}: {e}"


class Deduplicator(FilterStep):
    def __init__(self, config: dict):
        self.config = config
        self.hash_size = config.get("hash_size", 8)
        self.threshold = config.get("threshold", 5)
        self.seen_hashes = []  # (image_item, hash_obj) 튜플의 리스트

    def run(self, images: list[ImageItem]) -> list[ImageItem]:
        unique_images = []
        duplicates = 0

        print(f"[Deduplicator] Processing {len(images)} images with ProcessPoolExecutor (bypassing GIL)...")
        total = len(images)
        if total == 0:
            return unique_images

        # CPU 코어 수에 맞춰 워커 수 조정 (일반적으로 CPU 코어 수)
        max_workers = max(1, min(int(getattr(settings, "max_workers", 64)) // 4, 16))
        hash_results: list[tuple[ImageItem, imagehash.ImageHash | None, str | None]] = [None] * total

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_hash_worker, img_item.path, self.hash_size): idx
                for idx, img_item in enumerate(images)
            }
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                img_item = images[idx]
                try:
                    current_hash, error = future.result()
                except Exception as e:
                    current_hash, error = None, f"[Deduplicator] Error processing {img_item.path}: {e}"

                hash_results[idx] = (img_item, current_hash, error)
                completed += 1
                print(f"[Deduplicator] Checking {completed}/{total}...", end="\r", flush=True)

        print()
        for img_item, current_hash, error in hash_results:
            if error:
                print(error)
                continue

            is_duplicate = False
            for seen_item, seen_hash in self.seen_hashes:
                # 거리는 해밍 거리
                if current_hash - seen_hash <= self.threshold:
                    is_duplicate = True
                    duplicates += 1
                    # print(f"Duplicate found: {img_item.id} similar to {seen_item.id}")
                    break

            if not is_duplicate:
                self.seen_hashes.append((img_item, current_hash))
                unique_images.append(img_item)

        print(f"[Deduplicator] Removed {duplicates} duplicates. Kept {len(unique_images)} images.")
        return unique_images
