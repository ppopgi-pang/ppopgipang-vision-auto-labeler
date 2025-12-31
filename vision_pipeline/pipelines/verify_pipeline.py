import yaml
import json
import asyncio
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
            print(f"경고: 설정 파일 {config_file}을 찾을 수 없습니다. 기본값을 사용합니다.")
            self.config = {}
            
        self.verifier = LLMVerifier(self.config)
        self.store = MetadataStore()

    def run(self, detection_results: list[dict]) -> list[LabelResult]:
        """
        입력: DetectPipeline의 출력 (딕셔너리 리스트)
        출력: LabelResult 리스트
        """
        print(f"--- VerifyPipeline Start ({len(detection_results)} items) with async batch processing ---")

        # 모든 (crop_path, label, image_id) 수집
        crop_data = []
        for item in detection_results:
            image_id = item.get("image_id")
            crop_paths = item.get("crop_paths", [])
            bboxes = item.get("bboxes", [])

            if len(crop_paths) != len(bboxes):
                print(f"[VerifyPipeline] 경고: {image_id}의 crop/bbox 불일치. 건너뜁니다.")
                continue

            for i, crop_path in enumerate(crop_paths):
                bbox_label = bboxes[i]["label"]
                crop_data.append((crop_path, bbox_label, image_id))

        total_crops = len(crop_data)
        print(f"[VerifyPipeline] Total crops to verify: {total_crops}")

        if total_crops == 0:
            print("--- VerifyPipeline Complete. No crops to verify. ---")
            return []

        # 비동기 배치 검증 실행
        batch_size = self.config.get("batch_size", 16)  # 동시 API 요청 수
        final_results = []

        async def run_async():
            results = []
            for batch_start in range(0, total_crops, batch_size):
                batch_end = min(batch_start + batch_size, total_crops)
                batch = crop_data[batch_start:batch_end]

                # (crop_path, label) 튜플 리스트 생성
                crop_label_pairs = [(crop_path, label) for crop_path, label, _ in batch]

                # 배치 비동기 검증
                batch_results = await self.verifier.verify_batch_async(crop_label_pairs)

                # image_id 첨부
                for i, result in enumerate(batch_results):
                    result.image_id = batch[i][2]  # image_id
                    results.append(result)

                # 진행상황 출력
                processed = len(results)
                verified_count = len([r for r in results if r.verified])
                print(f"[VerifyPipeline] Verified {processed}/{total_crops} (confirmed: {verified_count})...", end="\r", flush=True)

            return results

        # asyncio.run()으로 비동기 코드 실행
        final_results = asyncio.run(run_async())

        print()
        # 결과 저장
        output_path = Path("data/artifacts/verification.json")
        self.store.save([r.to_dict() for r in final_results], output_path)

        verified_count = len([r for r in final_results if r.verified])
        print(f"--- VerifyPipeline Complete. Verified {verified_count}/{len(final_results)} items. ---")
        return final_results
