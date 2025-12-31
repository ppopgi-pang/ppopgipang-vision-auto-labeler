import yaml
from pathlib import Path
from pipelines.base import PipelineStep
from modules.filter.dedup import Deduplicator
from modules.filter.quality import QualityFilter
from modules.filter.classifier import Classifier
from modules.storage.metadata_store import MetadataStore

class FilterPipeline(PipelineStep):
    def __init__(self, config_path: str = "configs/filter.yaml"):
        project_root = Path(__file__).resolve().parent.parent
        config_file = project_root / config_path
        
        if config_file.exists():
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            print(f"경고: 설정 파일 {config_file}을 찾을 수 없습니다. 기본값을 사용합니다.")
            self.config = {}
        
        self.deduplicator = Deduplicator(self.config.get("dedup", {}))
        self.quality_filter = QualityFilter(self.config.get("quality", {}))
        self.classifier = Classifier(self.config.get("classifier", {}))
        self.store = MetadataStore()

    def run(self, images):
        print(f"--- FilterPipeline Start ({len(images)} images) ---")

        # 1. 중복 제거
        images = self.deduplicator.run(images)

        # 2. 품질 필터링
        images = self.quality_filter.run(images)

        # 3. 분류 (CLIP 존재 확인)
        images = self.classifier.run(images)

        # 4. 결과 저장
        output_path = Path("data/artifacts/filtered.json")
        self.store.save(images, output_path)
        
        print(f"--- FilterPipeline Complete ({len(images)} images kept) ---")
        return images
