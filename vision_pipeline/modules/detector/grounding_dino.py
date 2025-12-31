from ...domain.image import ImageItem
from ...domain.bbox import BoundingBox

class GroundingDinoDetector:
    def detect(self, image: ImageItem) -> list[BoundingBox]:
        print(f"[GroundingDinoDetector] Detecting in {image.path}")
        return []
