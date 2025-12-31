from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Job:
    keywords: List[str]
    target_class: str = "object"  # 주로 CLIP 검증에 사용
    limit: int = 100
    job_id: str = field(default="default_job")

    # 추적용 메타데이터
    status: str = "created"
