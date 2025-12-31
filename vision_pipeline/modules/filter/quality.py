from PIL import Image
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.filter.base import FilterStep
from domain.image import ImageItem
from config import settings

class QualityFilter(FilterStep):
    def __init__(self, config: dict):
        self.config = config
        self.min_width = config.get("min_width", 256)
        self.min_height = config.get("min_height", 256)
        self.max_aspect_ratio = config.get("max_aspect_ratio", 3.0)
        self.min_file_size_kb = config.get("min_file_size_kb", 5)

    def run(self, images: list[ImageItem]) -> list[ImageItem]:
        passed_images = []
        rejected = 0
        
        print(f"[QualityFilter] Processing {len(images)} images...")
        total = len(images)
        if total == 0:
            return passed_images

        max_workers = max(1, int(getattr(settings, "max_workers", 4)))
        results: list[tuple[ImageItem, str, str | None]] = [None] * total

        def check_one(img_item: ImageItem) -> tuple[str, str | None]:
            if not img_item.path or not Path(img_item.path).exists():
                return "missing", f"[QualityFilter] Path not found: {img_item.path}"

            path = Path(img_item.path)
            try:
                size_kb = os.path.getsize(path) / 1024
                if size_kb < self.min_file_size_kb:
                    return "rejected", None

                with Image.open(path) as pil_img:
                    width, height = pil_img.size

                    if width < self.min_width or height < self.min_height:
                        return "rejected", None

                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > self.max_aspect_ratio:
                        return "rejected", None

                return "passed", None
            except Exception as e:
                return "error", f"[QualityFilter] Error checking {path}: {e}"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(check_one, img_item): idx
                for idx, img_item in enumerate(images)
            }
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                img_item = images[idx]
                try:
                    status, message = future.result()
                except Exception as e:
                    status, message = "error", f"[QualityFilter] Error checking {img_item.path}: {e}"

                results[idx] = (img_item, status, message)
                completed += 1
                print(f"[QualityFilter] Checking {completed}/{total}...", end="\r", flush=True)

        print()
        for img_item, status, message in results:
            if status == "passed":
                passed_images.append(img_item)
                continue

            if status == "missing":
                if message:
                    print(message)
                continue

            if status == "error" and message:
                print(message)
                rejected += 1
                continue

            rejected += 1
        print(f"[QualityFilter] Rejected {rejected} low quality images. Kept {len(passed_images)} images.")
        return passed_images
