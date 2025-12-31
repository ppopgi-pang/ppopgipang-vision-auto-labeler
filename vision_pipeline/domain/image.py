from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

@dataclass
class ImageItem:
    id: str = field(default="")
    path: Optional[Path] = None
    source: str = ""
    url: Optional[str] = None
    keyword: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": str(self.path) if self.path else None,
            "source": self.source,
            "url": self.url,
            "keyword": self.keyword,
            "meta": self.meta
        }
