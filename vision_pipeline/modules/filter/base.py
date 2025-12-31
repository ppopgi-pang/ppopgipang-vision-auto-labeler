from abc import ABC, abstractmethod
from domain.image import ImageItem

class FilterStep(ABC):
    @abstractmethod
    def run(self, images: list[ImageItem]) -> list[ImageItem]:
        pass
