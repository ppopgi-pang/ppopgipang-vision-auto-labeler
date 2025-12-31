from PIL import Image
from pathlib import Path
import os
from modules.filter.base import FilterStep
from domain.image import ImageItem

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

        for idx, img_item in enumerate(images, start=1):
            print(f"[QualityFilter] Checking {idx}/{total}...", end="\r", flush=True)
            if not img_item.path or not Path(img_item.path).exists():
                print(f"[QualityFilter] Path not found: {img_item.path}")
                continue
                
            path = Path(img_item.path)

            # 파일 크기 확인
            try:
                size_kb = os.path.getsize(path) / 1024
                if size_kb < self.min_file_size_kb:
                    # print(f"Rejected {img_item.id}: size {size_kb:.1f}KB < {self.min_file_size_kb}KB")
                    rejected += 1
                    continue

                # 차원 및 종횡비 확인
                with Image.open(path) as pil_img:
                    width, height = pil_img.size

                    if width < self.min_width or height < self.min_height:
                        # print(f"Rejected {img_item.id}: {width}x{height} < min {self.min_width}x{self.min_height}")
                        rejected += 1
                        continue

                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > self.max_aspect_ratio:
                        # print(f"Rejected {img_item.id}: AR {aspect_ratio:.2f} > {self.max_aspect_ratio}")
                        rejected += 1
                        continue
                        
                passed_images.append(img_item)
                
            except Exception as e:
                print(f"[QualityFilter] Error checking {path}: {e}")
                rejected += 1

        print()
        print(f"[QualityFilter] Rejected {rejected} low quality images. Kept {len(passed_images)} images.")
        return passed_images
