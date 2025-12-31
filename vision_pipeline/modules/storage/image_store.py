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
                # 디렉토리 생성: settings.output_dir/raw/{keyword}
                keyword_slug = item.keyword.replace(" ", "_") if item.keyword else "unknown"
                save_dir = os.path.join(settings.output_dir, "raw", keyword_slug)
                os.makedirs(save_dir, exist_ok=True)

                # 파일명 생성
                # ID가 설정되지 않은 경우 생성
                if not item.id:
                    item.id = str(uuid.uuid4())

                filename = f"{item.id}.jpg"  # 현재는 jpg로 가정, 또는 헤더에서 감지?
                # 더 나은 접근 방식은 content-type을 확인하거나 그대로 저장하고 나중에 감지하는 것.
                # 대부분이 이미지이므로 단순화를 위해 .jpg 사용. 
                
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
