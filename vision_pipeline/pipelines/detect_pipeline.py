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
            batch_items = [img for img in images[batch_start:batch_end] if img.path]

            if not batch_items:
                continue

            # 배치 탐지 실행
            batch_bboxes = self.detector.detect_batch(batch_items)

            # 결과 처리
            for img_item, bboxes in zip(batch_items, batch_bboxes):
                crop_paths = []
                if bboxes and self.save_crops:
                    for idx, bbox in enumerate(bboxes):
                        # 고유 크롭 경로 정의: data/crops/{label}/{image_id}_{idx}.jpg
                        # 경로를 위한 레이블 정리
                        label_clean = "".join(c for c in bbox.label if c.isalnum() or c in (' ', '_', '-')).strip()

                        crop_filename = f"{img_item.id}_{idx}.jpg"
                        crop_path = Path("data/crops") / label_clean / crop_filename

                        success = crop_image(img_item.path, bbox, crop_path, padding=self.crop_padding)
                        if success:
                            crop_paths.append(str(crop_path))

                # 결과 항목 생성
                result_entry = {
                    "image_id": img_item.id,
                    "original_path": str(img_item.path),
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

            # 진행상황 출력
            processed = min(batch_end, total)
            print(f"[DetectPipeline] Processed {processed}/{total}...", end="\r", flush=True)

        print()
        # 결과 저장
        output_path = Path("data/artifacts/bboxes.json")
        self.store.save(results, output_path)

        print(f"--- DetectPipeline Complete. Processed {len(images)} images. Results saved to {output_path} ---")
        return results
