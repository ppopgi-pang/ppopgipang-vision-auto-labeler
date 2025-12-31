from abc import ABC, abstractmethod
from typing import Any, List

class PipelineStep(ABC):
    @abstractmethod
    def run(self, items: Any) -> Any:
        pass
