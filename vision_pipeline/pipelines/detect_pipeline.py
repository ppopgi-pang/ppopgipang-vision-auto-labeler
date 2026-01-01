import time
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from tqdm import tqdm
from pipelines.base import PipelineStep
from modules.detector.yolo_world import YoloDetector
from modules.detector.bbox_utils import crop_image, crop_image_to_pil, draw_bboxes
from modules.storage.metadata_store import MetadataStore
from modules.llm.labeler import VLMLabeler
from modules.clip.candidate_generator import CLIPCandidateGenerator
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
        self.use_vlm_labeler = self.config.get("use_vlm_labeler", False)
        self.labeler_fallback_label = self.config.get("labeler_fallback_label", "unknown")
        self.save_annotated = self.config.get("save_annotated", False)
        self.annotated_dir = self.config.get("annotated_dir", "data/annotated")
        annotated_color = self.config.get("annotated_box_color", [0, 255, 0])
        if isinstance(annotated_color, (list, tuple)) and len(annotated_color) == 3:
            self.annotated_box_color = tuple(int(c) for c in annotated_color)
        else:
            self.annotated_box_color = (0, 255, 0)
        self.annotated_line_width = int(self.config.get("annotated_line_width", 2))
        self.annotated_font_size = int(self.config.get("annotated_font_size", 14))
        self.annotated_show_confidence = bool(self.config.get("annotated_show_confidence", True))
        self.labeler = None
        self.labeler_rate_limit_delay = 0.0
        self.api_semaphore = None
        self.force_fallback_label = False
        self.clip_candidate_generator = None
        self.clip_top1_threshold = None
        self.clip_semaphore = None
        if self.use_vlm_labeler:
            labeler_config = self.config.get("labeler", {})
            self.labeler_rate_limit_delay = float(labeler_config.get("rate_limit_delay", 0.0))
            api_max_concurrent = int(labeler_config.get("api_max_concurrent", 1))
            self.api_semaphore = Semaphore(api_max_concurrent)
            self.labeler = VLMLabeler(labeler_config)
            if not self.labeler.is_available():
                print("[DetectPipeline] VLM 라벨러를 사용할 수 없습니다. 기본 라벨을 사용합니다.")
                self.labeler = None
                self.force_fallback_label = True

        clip_config = self.config.get("clip_candidate", {})
        if bool(clip_config.get("enabled", False)):
            self.clip_top1_threshold = float(clip_config.get("top1_threshold", 0.55))
            clip_max_concurrent = int(clip_config.get("max_concurrent", 1))
            self.clip_semaphore = Semaphore(max(1, clip_max_concurrent))
            self.clip_candidate_generator = CLIPCandidateGenerator(clip_config)
            if not self.clip_candidate_generator.is_available():
                print("[DetectPipeline] CLIP 후보 생성기를 사용할 수 없습니다. VLM 라벨링을 건너뜁니다.")
                self.clip_candidate_generator = None

    def _process_single_crop(self, img_path, bbox, crop_idx, img_id):
        """단일 크롭을 처리: 생성 → 라벨링 → 저장 (스레드 안전)"""
        label = bbox.label
        crop_img = None
        labeler_confidence = None
        crop_path_str = None
        clip_candidates = None
        clip_top1_score = None

        try:
            # 1. 크롭 생성
            if self.force_fallback_label:
                label = self.labeler_fallback_label

            if self.labeler or self.save_crops:
                crop_img = crop_image_to_pil(img_path, bbox, padding=self.crop_padding)

            # 2. CLIP 후보 생성
            if self.clip_candidate_generator and crop_img and not self.force_fallback_label:
                with self.clip_semaphore:
                    clip_candidates, clip_top1_score = self.clip_candidate_generator.get_candidates(crop_img)

            # 3. VLM 라벨링 (CLIP 후보 기반, 세마포어로 API 호출 제어)
            if self.labeler and crop_img and clip_candidates:
                candidate_labels = [c["label"] for c in clip_candidates if c.get("label")]
                if not candidate_labels:
                    label = self.labeler_fallback_label
                elif clip_top1_score is not None and self.clip_top1_threshold is not None and clip_top1_score < self.clip_top1_threshold:
                    label = self.labeler_fallback_label
                else:
                    with self.api_semaphore:
                        label, labeler_confidence = self.labeler.label_image(crop_img, candidate_labels)
                        if self.labeler_rate_limit_delay > 0:
                            time.sleep(self.labeler_rate_limit_delay)
            elif crop_img is None and not self.force_fallback_label:
                label = self.labeler_fallback_label
            elif not self.force_fallback_label and self.labeler and crop_img and not clip_candidates:
                label = self.labeler_fallback_label

            # 4. 크롭 저장
            if self.save_crops:
                label_clean = "".join(c for c in label if c.isalnum() or c in (" ", "_", "-")).strip()
                if not label_clean:
                    label_clean = self.labeler_fallback_label

                crop_filename = f"{img_id}_{crop_idx}.jpg"
                crop_path = Path("data/crops") / label_clean / crop_filename

                # 디렉토리 먼저 생성 (crop_img 여부와 관계없이)
                crop_path.parent.mkdir(parents=True, exist_ok=True)

                if crop_img:
                    crop_img.save(crop_path)
                    crop_path_str = str(crop_path)
                else:
                    success = crop_image(img_path, bbox, crop_path, padding=self.crop_padding)
                    if success:
                        crop_path_str = str(crop_path)

        except Exception as e:
            print(f"\n[DetectPipeline] 크롭 처리 실패 {img_path} (idx={crop_idx}): {e}")
        finally:
            if crop_img:
                try:
                    crop_img.close()
                except Exception:
                    pass

        return crop_path_str, labeler_confidence, label, clip_candidates, clip_top1_score

    def run(self, images: list[ImageItem]) -> list[dict]:
        """
        입력: ImageItem 리스트
        출력: {image_id, path, bboxes: [BoundingBox], crop_paths: [Path]} 키를 가진 딕셔너리 리스트
        """
        print(f"--- DetectPipeline 시작 ({len(images)}개 이미지) 배치 처리 방식 ---")

        results = []
        total = len(images)
        batch_size = self.config.get("batch_size", 16)  # YOLO 배치 크기 (8-16 권장)

        # 배치 단위로 처리 (position=1: 전체 프로그레스바 아래, leave=False: 완료 후 제거)
        with tqdm(total=total, desc="객체 탐지", unit="img", position=1, leave=False) as pbar:
            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch_items = images[batch_start:batch_end]

                # path가 있는 항목과 인덱스 매핑
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
                        print(f"\n[DetectPipeline] 배치 검출 오류: {e}, 개별 처리로 전환")
                        # 에러 복구: 개별 처리
                        batch_bboxes = []
                        for item in valid_items:
                            try:
                                bboxes = self.detector.detect(item)
                                batch_bboxes.append(bboxes)
                            except Exception as e2:
                                print(f"\n[DetectPipeline] 개별 검출 실패 {item.path}: {e2}")
                                batch_bboxes.append([])
                else:
                    batch_bboxes = []

                # 결과 처리: 모든 이미지에 대해 결과 생성 (path 없는 것도 포함)
                bbox_map = dict(zip(valid_indices, batch_bboxes))

                # 배치 내 모든 crops를 병렬 처리
                total_crops_in_batch = 0
                for idx, img_item in enumerate(batch_items):
                    bboxes = bbox_map.get(idx, [])
                    crop_paths = []
                    labeler_confidences = [None] * len(bboxes)
                    clip_candidates_list = [None] * len(bboxes)
                    clip_top1_scores = [None] * len(bboxes)

                    if bboxes and img_item.path and (self.save_crops or self.labeler or self.force_fallback_label):
                        # 병렬 처리: 모든 crops를 ThreadPoolExecutor로 처리
                        max_workers = min(len(bboxes), 10)  # 최대 10개 워커
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = {
                                executor.submit(
                                    self._process_single_crop,
                                    img_item.path,
                                    bbox,
                                    crop_idx,
                                    img_item.id
                                ): crop_idx
                                for crop_idx, bbox in enumerate(bboxes)
                            }

                            for future in as_completed(futures):
                                crop_idx = futures[future]
                                try:
                                    crop_path_str, labeler_confidence, label, clip_candidates, clip_top1_score = future.result()
                                    if crop_path_str:
                                        crop_paths.append(crop_path_str)
                                    labeler_confidences[crop_idx] = labeler_confidence
                                    clip_candidates_list[crop_idx] = clip_candidates
                                    clip_top1_scores[crop_idx] = clip_top1_score
                                    # 메인 스레드에서 bbox.label 업데이트 (스레드 안전)
                                    if label:
                                        bboxes[crop_idx].label = label
                                except Exception as e:
                                    print(f"\n[DetectPipeline] 크롭 처리 오류: {e}")

                        total_crops_in_batch += len(bboxes)
                    annotated_path = None
                    if self.save_annotated and img_item.path and bboxes:
                        image_stem = img_item.id or Path(img_item.path).stem
                        annotated_file = f"{image_stem}.jpg"
                        annotated_output = Path(self.annotated_dir) / annotated_file
                        success = draw_bboxes(
                            img_item.path,
                            bboxes,
                            annotated_output,
                            color=self.annotated_box_color,
                            width=self.annotated_line_width,
                            font_size=self.annotated_font_size,
                            show_confidence=self.annotated_show_confidence,
                        )
                        if success:
                            annotated_path = str(annotated_output)

                    # 결과 항목 생성 (모든 이미지에 대해)
                    result_entry = {
                        "image_id": img_item.id,
                        "original_path": str(img_item.path) if img_item.path else None,
                        "bboxes": [
                            {
                                "label": b.label,
                                "confidence": b.confidence,
                                "xyxy": b.xyxy,
                                "labeler_confidence": labeler_confidences[i],
                                "clip_candidates": clip_candidates_list[i],
                                "clip_top1_score": clip_top1_scores[i],
                            } for i, b in enumerate(bboxes)
                        ],
                        "crop_paths": crop_paths,
                        "annotated_path": annotated_path,
                    }
                    results.append(result_entry)

                # 프로그레스바 업데이트
                detected_count = sum(1 for r in results if r["bboxes"])
                pbar.set_postfix_str(f"검출: {detected_count}")
                pbar.update(len(batch_items))

        print()
        # 결과 저장
        output_path = Path("data/artifacts/bboxes.json")
        self.store.save(results, output_path)

        detected_count = sum(1 for r in results if r["bboxes"])
        print(f"--- DetectPipeline 완료. {len(images)}개 이미지 처리, {detected_count}개에서 검출됨. 결과 저장: {output_path} ---")
        return results
