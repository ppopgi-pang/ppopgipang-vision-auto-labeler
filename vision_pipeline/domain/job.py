from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Job:
    keywords: List[str]
    target_class: str = "object" # For CLIP verification mainly
    limit: int = 100
    job_id: str = field(default="default_job")
    
    # Metadata for tracking
    status: str = "created"
