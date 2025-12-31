from dataclasses import dataclass

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    @property
    def xyxy(self):
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1
