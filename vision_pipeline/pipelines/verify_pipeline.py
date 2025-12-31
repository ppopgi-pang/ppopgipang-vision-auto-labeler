import yaml
import json
import asyncio
import time
from pathlib import Path
from pipelines.base import PipelineStep
from modules.llm.verifier import LLMVerifier
from modules.storage.metadata_store import MetadataStore
from domain.label import LabelResult

# Colab/Jupyter 환경을 위한 nest_asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
    NESTED_ASYNCIO_AVAILABLE = True
except ImportError:
    NESTED_ASYNCIO_AVAILABLE = False

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
        print(f"--- VerifyPipeline 시작 ({len(detection_results)}개 항목) 비동기 배치 처리 방식 ---")

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
        print(f"[VerifyPipeline] 검증할 크롭 총 개수: {total_crops}")

        if total_crops == 0:
            print("--- VerifyPipeline 완료. 검증할 크롭이 없습니다. ---")
            return []

        # Rate limit 고려한 배치 크기
        batch_size = self.config.get("batch_size", 8)  # 8개로 축소 (OpenAI rate limit 고려)
        rate_limit_delay = self.config.get("rate_limit_delay", 1.0)  # 배치 간 지연 (초)
        final_results = []

        async def run_async():
            results = []
            num_batches = (total_crops + batch_size - 1) // batch_size

            for batch_idx, batch_start in enumerate(range(0, total_crops, batch_size), start=1):
                batch_end = min(batch_start + batch_size, total_crops)
                batch = crop_data[batch_start:batch_end]

                # (crop_path, label) 튜플 리스트 생성
                crop_label_pairs = [(crop_path, label) for crop_path, label, _ in batch]

                try:
                    # 배치 비동기 검증
                    batch_results = await self.verifier.verify_batch_async(crop_label_pairs)

                    # image_id 첨부
                    for i, result in enumerate(batch_results):
                        result.image_id = batch[i][2]  # image_id
                        results.append(result)

                except Exception as e:
                    print(f"\n[VerifyPipeline] 배치 {batch_idx} 오류: {e}, 개별 처리로 전환")
                    # 에러 복구: 개별 처리
                    for crop_path, label, image_id in batch:
                        try:
                            result = self.verifier.verify_image(crop_path, label)
                            result.image_id = image_id
                            results.append(result)
                        except Exception as e2:
                            print(f"\n[VerifyPipeline] 개별 검증 실패 {crop_path}: {e2}")
                            results.append(LabelResult(
                                image_id=image_id,
                                crop_path=str(crop_path),
                                verified=False,
                                label=label,
                                reason=f"Error: {e2}",
                                confidence=0.0
                            ))

                # 진행상황 출력
                processed = len(results)
                verified_count = len([r for r in results if r.verified])
                print(f"[VerifyPipeline] 검증됨 {processed}/{total_crops} (확인됨: {verified_count}, 배치: {batch_idx}/{num_batches})...", end="\r", flush=True)

                # Rate limit 방지를 위한 지연
                if batch_idx < num_batches and rate_limit_delay > 0:
                    await asyncio.sleep(rate_limit_delay)

            return results

        # Colab/Jupyter 환경에서 asyncio.run() 충돌 방지
        try:
            # 이미 실행 중인 이벤트 루프가 있는지 확인
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Jupyter/Colab 환경: nest_asyncio 사용
                if NESTED_ASYNCIO_AVAILABLE:
                    final_results = asyncio.run(run_async())
                else:
                    # nest_asyncio 없으면 동기 처리로 fallback
                    print("[VerifyPipeline] 경고: nest_asyncio를 사용할 수 없습니다. 동기 처리로 전환합니다.")
                    final_results = self._run_sync_fallback(crop_data)
            else:
                # 일반 환경
                final_results = asyncio.run(run_async())
        except RuntimeError as e:
            # asyncio.run() 실패 시 동기 처리로 fallback
            print(f"[VerifyPipeline] 비동기 실행 실패: {e}, 동기 처리로 전환합니다.")
            final_results = self._run_sync_fallback(crop_data)

        print()
        # 결과 저장
        output_path = Path("data/artifacts/verification.json")
        self.store.save([r.to_dict() for r in final_results], output_path)

        verified_count = len([r for r in final_results if r.verified])
        print(f"--- VerifyPipeline 완료. {verified_count}/{len(final_results)}개 항목 검증됨. ---")
        return final_results

    def _run_sync_fallback(self, crop_data: list[tuple[str, str, str]]) -> list[LabelResult]:
        """동기 처리 fallback (asyncio 실패 시)"""
        results = []
        total = len(crop_data)

        for idx, (crop_path, label, image_id) in enumerate(crop_data, start=1):
            try:
                result = self.verifier.verify_image(crop_path, label)
                result.image_id = image_id
                results.append(result)
            except Exception as e:
                print(f"\n[VerifyPipeline] 동기 검증 실패 {crop_path}: {e}")
                results.append(LabelResult(
                    image_id=image_id,
                    crop_path=str(crop_path),
                    verified=False,
                    label=label,
                    reason=f"Error: {e}",
                    confidence=0.0
                ))

            verified_count = len([r for r in results if r.verified])
            print(f"[VerifyPipeline] 검증됨 {idx}/{total} (확인됨: {verified_count})...", end="\r", flush=True)

        return results
