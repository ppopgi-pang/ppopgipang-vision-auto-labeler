from abc import ABC, abstractmethod
from typing import List, Any
from domain.image import ImageItem

class Crawler(ABC):
    @abstractmethod
    def fetch(self, keywords: List[str]) -> List[ImageItem]:
        pass
