import yaml
import json
from pathlib import Path
from pipelines.base import PipelineStep
from modules.llm.verifier import LLMVerifier
from modules.storage.metadata_store import MetadataStore
from domain.label import LabelResult

class VerifyPipeline(PipelineStep):
    def __init__(self, config_path: str = "configs/llm.yaml"):
        project_root = Path(__file__).resolve().parent.parent
        config_file = project_root / config_path
        
        if config_file.exists():
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            print(f"Warning: Config file {config_file} not found. Using defaults.")
            self.config = {}
            
        self.verifier = LLMVerifier(self.config)
        self.store = MetadataStore()

    def run(self, detection_results: list[dict]) -> list[LabelResult]:
        """
        입력: DetectPipeline의 출력 (딕셔너리 리스트)
        출력: LabelResult 리스트
        """
        print(f"--- VerifyPipeline Start ({len(detection_results)} items) ---")

        final_results = []

        for item in detection_results:
            image_id = item.get("image_id")
            crop_paths = item.get("crop_paths", [])
            bboxes = item.get("bboxes", [])

            # bboxes와 crop_paths가 인덱스로 정렬되어 있다고 가정
            # 또는 crop_paths만 반복할 수도 있음.
            # 하지만 bbox에서 레이블이 필요함.

            if len(crop_paths) != len(bboxes):
                print(f"[VerifyPipeline] Warning: Mismatch crops/bboxes for {image_id}. Skipping.")
                continue

            for i, crop_path in enumerate(crop_paths):
                bbox_label = bboxes[i]["label"]

                # 검증 실행
                result = self.verifier.verify_image(crop_path, label=bbox_label)
                result.image_id = image_id  # 부모 ID 첨부

                if result.verified:
                    print(f"[VerifyPipeline] Verified {image_id} as {bbox_label} (Conf: {result.confidence})")
                else:
                    # print(f"[VerifyPipeline] Rejected {image_id} as {bbox_label}: {result.reason}")
                    pass

                final_results.append(result)

        # 결과 저장
        output_path = Path("data/artifacts/verification.json")
        self.store.save([r.to_dict() for r in final_results], output_path)
            
        print(f"--- VerifyPipeline Complete. Verified {len([r for r in final_results if r.verified])}/{len(final_results)} items. ---")
        return final_results
