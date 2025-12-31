import json
from pathlib import Path
from typing import List, Any
from domain.image import ImageItem

class MetadataStore:
    def save(self, data: Any, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[MetadataStore] Saving metadata to {path}")
        
        # Determine if data is list of ImageItem or just list of dict or dict
        serializable_data = data
        if isinstance(data, list):
            if data and isinstance(data[0], ImageItem):
                serializable_data = [item.to_dict() for item in data]
        elif isinstance(data, ImageItem):
            serializable_data = data.to_dict()
            
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

    def load(self, path: str | Path) -> Any:
        path = Path(path)
        if not path.exists():
            print(f"[MetadataStore] File not found: {path}")
            return None
            
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
