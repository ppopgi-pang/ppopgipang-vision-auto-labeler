from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import torch
from domain.image import ImageItem
from domain.bbox import BoundingBox

class YoloDetector:
    """YOLO 모델을 사용한 객체 검출 클래스"""

    def __init__(self, config: dict):
        self.config = config
        self.model_path = config.get("model_path", "yolov8n.pt")
        self.conf_threshold = config.get("conf_threshold", 0.5)
        self.classes = config.get("classes", None)  # 필터링할 클래스 인덱스 리스트, None = 모두
        self.device = config.get("device", "auto")

        # GPU/CUDA 장치 자동 감지
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # 장치 가용성 확인 및 fallback
        if self.device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                print("[YoloDetector] CUDA not available, falling back to MPS")
                self.device = "mps"
            else:
                print("[YoloDetector] CUDA not available, falling back to CPU")
                self.device = "cpu"

        if self.device == "mps" and not torch.backends.mps.is_available():
            if torch.cuda.is_available():
                print("[YoloDetector] MPS not available, falling back to CUDA")
                self.device = "cuda"
            else:
                print("[YoloDetector] MPS not available, falling back to CPU")
                self.device = "cpu"

        print(f"[YoloDetector] {self.model_path}에서 모델 로딩 중... (device: {self.device})")
        try:
            self.model = YOLO(self.model_path)
            # YOLO 모델을 지정된 장치로 이동
            self.model.to(self.device)
            print(f"[YoloDetector] 모델 로딩 완료. (device: {self.device})")
        except Exception as e:
            print(f"[YoloDetector] 모델 로딩 실패: {e}")
            raise e

    def detect(self, image_item: ImageItem) -> list[BoundingBox]:
        """단일 이미지에서 객체 검출"""
        if not image_item.path:
            return []

        try:
            # YOLO 추론 실행 (GPU 사용)
            results = self.model.predict(
                source=str(image_item.path),
                conf=self.conf_threshold,
                classes=self.classes,
                device=self.device,
                verbose=False
            )

            bboxes = []
            for result in results:
                for box in result.boxes:
                    # box.xyxy는 [x1, y1, x2, y2] 형식
                    coords = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    label = self.model.names[cls_id]

                    bboxes.append(BoundingBox(
                        x1=coords[0],
                        y1=coords[1],
                        x2=coords[2],
                        y2=coords[3],
                        confidence=conf,
                        label=label
                    ))

            return bboxes

        except Exception as e:
            print(f"[YoloDetector] 검출 오류 {image_item.path}: {e}")
            return []

    def detect_batch(self, image_items: list[ImageItem]) -> list[list[BoundingBox]]:
        """
        배치 이미지 탐지 (GPU 효율성 향상)
        Returns: 각 이미지에 대한 BoundingBox 리스트의 리스트
        """
        if not image_items:
            return []

        # 유효한 이미지만 필터링
        valid_items = [item for item in image_items if item.path]
        if not valid_items:
            return [[] for _ in image_items]

        try:
            # 배치 추론 실행 (경로 리스트 전달, GPU 사용)
            sources = [str(item.path) for item in valid_items]
            results = self.model.predict(
                source=sources,
                conf=self.conf_threshold,
                classes=self.classes,
                device=self.device,
                verbose=False
            )

            # 각 이미지에 대한 결과 파싱
            all_bboxes = []
            for result in results:
                bboxes = []
                for box in result.boxes:
                    coords = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    label = self.model.names[cls_id]

                    bboxes.append(BoundingBox(
                        x1=coords[0],
                        y1=coords[1],
                        x2=coords[2],
                        y2=coords[3],
                        confidence=conf,
                        label=label
                    ))
                all_bboxes.append(bboxes)

            return all_bboxes

        except Exception as e:
            print(f"[YoloDetector] 배치 검출 오류: {e}")
            return [[] for _ in valid_items]
