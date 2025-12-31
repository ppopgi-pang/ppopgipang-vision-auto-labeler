import imagehash
from PIL import Image
from pathlib import Path
from modules.filter.base import FilterStep
from domain.image import ImageItem

class Deduplicator(FilterStep):
    def __init__(self, config: dict):
        self.config = config
        self.hash_size = config.get("hash_size", 8)
        self.threshold = config.get("threshold", 5)
        self.seen_hashes = []  # (image_item, hash_obj) 튜플의 리스트

    def run(self, images: list[ImageItem]) -> list[ImageItem]:
        unique_images = []
        duplicates = 0

        print(f"[Deduplicator] Processing {len(images)} images...")

        for img_item in images:
            if not img_item.path or not Path(img_item.path).exists():
                print(f"[Deduplicator] specific path not found: {img_item.path}")
                continue

            try:
                # 이미지 열기
                with Image.open(img_item.path) as pil_img:
                    # 해시 계산
                    current_hash = imagehash.phash(pil_img, hash_size=self.hash_size)

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

            except Exception as e:
                print(f"[Deduplicator] Error processing {img_item.path}: {e}")
                
        print(f"[Deduplicator] Removed {duplicates} duplicates. Kept {len(unique_images)} images.")
        return unique_images
