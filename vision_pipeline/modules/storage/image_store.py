from domain.image import ImageItem

import requests
import uuid
import os
from pathlib import Path
from config import settings

class ImageStore:
    def save_raw(self, images: list[ImageItem]):
        print(f"[ImageStore] Saving {len(images)} raw images")
        
        for item in images:
            if item.path:
                continue
                
            if not item.url:
                continue
                
            try:
                # Create directory: settings.output_dir/raw/{keyword}
                keyword_slug = item.keyword.replace(" ", "_") if item.keyword else "unknown"
                save_dir = os.path.join(settings.output_dir, "raw", keyword_slug)
                os.makedirs(save_dir, exist_ok=True)
                
                # Generate filename
                # If ID is not set, generate one
                if not item.id:
                    item.id = str(uuid.uuid4())
                    
                filename = f"{item.id}.jpg" # assuming jpg for now, or detect from header?
                # A better approach is to check content-type or just save as is and detect later.
                # using .jpg for simplicity as most are images. 
                
                filepath = os.path.join(save_dir, filename)
                
                response = requests.get(item.url, timeout=10)
                if response.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    
                    item.path = Path(filepath)
                    # print(f"Saved {filepath}")
                else:
                    print(f"Failed to download {item.url}: {response.status_code}")
                    
            except Exception as e:
                print(f"Error saving image {item.url}: {e}")
