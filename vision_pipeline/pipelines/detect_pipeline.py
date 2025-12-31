import yaml
from pathlib import Path
from pipelines.base import PipelineStep
from modules.detector.yolo_world import YoloDetector
from modules.detector.bbox_utils import crop_image
from modules.storage.metadata_store import MetadataStore
from domain.image import ImageItem

class DetectPipeline(PipelineStep):
    def __init__(self, config_path: str = "configs/detector.yaml"):
        project_root = Path(__file__).resolve().parent.parent
        config_file = project_root / config_path
        
        if config_file.exists():
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            print(f"경고: 설정 파일 {config_file}을 찾을 수 없습니다. 기본값을 사용합니다.")
            self.config = {}
            
        self.detector = YoloDetector(self.config)
        self.store = MetadataStore()
        
        self.save_crops = self.config.get("save_crops", True)
        self.crop_padding = self.config.get("crop_padding", 0)

    def run(self, images: list[ImageItem]) -> list[dict]:
        """
        입력: ImageItem 리스트
        출력: {image_id, path, bboxes: [BoundingBox], crop_paths: [Path]} 키를 가진 딕셔너리 리스트
        """
        print(f"--- DetectPipeline Start ({len(images)} images) with batch processing ---")

        results = []
        total = len(images)
        batch_size = self.config.get("batch_size", 16)  # YOLO 배치 크기 (8-16 권장)

        # 배치 단위로 처리
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_items = images[batch_start:batch_end]

            # 문제 4 해결: path가 있는 항목과 인덱스 매핑
            valid_items = []
            valid_indices = []
            for idx, img in enumerate(batch_items):
                if img.path:
                    valid_items.append(img)
                    valid_indices.append(idx)

            # 배치 탐지 실행
            if valid_items:
                try:
                    batch_bboxes = self.detector.detect_batch(valid_items)
                except Exception as e:
                    print(f"\n[DetectPipeline] Batch detection error: {e}, falling back to individual processing")
                    # 문제 8 해결: 에러 복구
                    batch_bboxes = []
                    for item in valid_items:
                        try:
                            bboxes = self.detector.detect(item)
                            batch_bboxes.append(bboxes)
                        except Exception as e2:
                            print(f"\n[DetectPipeline] Individual detection failed {item.path}: {e2}")
                            batch_bboxes.append([])
            else:
                batch_bboxes = []

            # 결과 처리: 모든 이미지에 대해 결과 생성 (path 없는 것도 포함)
            bbox_map = dict(zip(valid_indices, batch_bboxes))

            for idx, img_item in enumerate(batch_items):
                bboxes = bbox_map.get(idx, [])  # path 없으면 빈 리스트
                crop_paths = []

                if bboxes and self.save_crops and img_item.path:
                    for crop_idx, bbox in enumerate(bboxes):
                        # 고유 크롭 경로 정의: data/crops/{label}/{image_id}_{idx}.jpg
                        # 경로를 위한 레이블 정리
                        label_clean = "".join(c for c in bbox.label if c.isalnum() or c in (' ', '_', '-')).strip()

                        crop_filename = f"{img_item.id}_{crop_idx}.jpg"
                        crop_path = Path("data/crops") / label_clean / crop_filename

                        try:
                            success = crop_image(img_item.path, bbox, crop_path, padding=self.crop_padding)
                            if success:
                                crop_paths.append(str(crop_path))
                        except Exception as e:
                            print(f"\n[DetectPipeline] Crop failed for {img_item.path}: {e}")

                # 결과 항목 생성 (모든 이미지에 대해)
                result_entry = {
                    "image_id": img_item.id,
                    "original_path": str(img_item.path) if img_item.path else None,
                    "bboxes": [
                        {
                            "label": b.label,
                            "confidence": b.confidence,
                            "xyxy": b.xyxy
                        } for b in bboxes
                    ],
                    "crop_paths": crop_paths
                }
                results.append(result_entry)

            # 진행상황 출력 (문제 9 해결: 정확한 진행률)
            processed = min(batch_end, total)
            detected_count = sum(1 for r in results if r["bboxes"])
            print(f"[DetectPipeline] Processed {processed}/{total} (detected: {detected_count})...", end="\r", flush=True)

        print()
        # 결과 저장
        output_path = Path("data/artifacts/bboxes.json")
        self.store.save(results, output_path)

        detected_count = sum(1 for r in results if r["bboxes"])
        print(f"--- DetectPipeline Complete. Processed {len(images)} images, detected in {detected_count}. Results saved to {output_path} ---")
        return results
