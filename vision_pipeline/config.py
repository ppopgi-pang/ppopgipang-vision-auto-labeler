from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # 일반 파이프라인 설정
    output_dir: str = "data"
    max_workers: int = 64  # T4 GPU 환경에 최적화 (다운로드, I/O 병렬화)
    pipeline_stages: List[str] = ["crawl", "filter", "detect", "verify"]

    # Google 크롤러 설정
    google_api_key: Optional[str] = Field(None, description="Google Custom Search API Key")
    google_cx: Optional[str] = Field(None, description="Google Custom Search CX")
    google_num_results: int = 100

    # Naver 크롤러 설정
    naver_client_id: Optional[str] = Field(None, description="Naver Search Client ID")
    naver_client_secret: Optional[str] = Field(None, description="Naver Search Client Secret")
    naver_num_results: int = 100

    # 필터 설정
    filter_dedup_method: str = "phash"
    filter_dedup_threshold: int = 5
    filter_min_width: int = 256
    filter_min_height: int = 256
    filter_min_face_size: int = 50

    # Detector (YOLO) 설정
    # 예상되는 필요에 따라 추가됨, 이전 yaml에는 엄격히 포함되지 않음
    yolo_model_path: str = "yolov8n.pt"

    # LLM / 검증 설정
    classifier_model: str = "openai/clip-vit-base-patch32"
    classifier_positive_prompts: List[str] = ["an image of hatsune miku"]
    classifier_negative_prompts: List[str] = ["text", "low quality", "blurred"]
    classifier_threshold: float = 0.25

settings = Settings()
