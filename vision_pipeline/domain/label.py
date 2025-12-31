from dataclasses import dataclass
from domain.bbox import BoundingBox
from typing import Optional

@dataclass
class LabelResult:
    image_id: str
    crop_path: str
    verified: bool
    label: str
    reason: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self):
        return {
            "image_id": self.image_id,
            "crop_path": self.crop_path,
            "verified": self.verified,
            "label": self.label,
            "reason": self.reason,
            "confidence": self.confidence
        }
