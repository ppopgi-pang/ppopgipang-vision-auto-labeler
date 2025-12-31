import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from modules.filter.base import FilterStep
from domain.image import ImageItem
from PIL import UnidentifiedImageError

class Classifier(FilterStep):
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.config = config
        self.model_name = config.get("model_name", "openai/clip-vit-base-patch32")
        self.threshold = config.get("threshold", 0.2)
        self.device = config.get("device", "auto")

        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        if self.device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                print("[Classifier] CUDA not available, falling back to MPS")
                self.device = "mps"
            else:
                print("[Classifier] CUDA not available, falling back to CPU")
                self.device = "cpu"

        if self.device == "mps" and not torch.backends.mps.is_available():
            if torch.cuda.is_available():
                print("[Classifier] MPS not available, falling back to CUDA")
                self.device = "cuda"
            else:
                print("[Classifier] MPS not available, falling back to CPU")
                self.device = "cpu"

        print(f"[Classifier] Loading CLIP model: {self.model_name} on {self.device}...")
        try:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            print("[Classifier] Model loaded successfully.")
        except Exception as e:
            print(f"[Classifier] Failed to load model: {e}")
            raise e

    def predict(self, image: Image.Image, text: list[str]) -> dict[str, float]:
        """
        {텍스트: 확률} 딕셔너리를 반환 (단일 이미지)
        """
        try:
            inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # 이미지-텍스트 유사도 점수
                probs = logits_per_image.softmax(dim=1)  # 제공된 클래스 간 확률이 필요한 경우 softmax 사용

            # 결과를 딕셔너리로 변환
            result = {text[i]: probs[0][i].item() for i in range(len(text))}
            return result
        except Exception as e:
            print(f"[Classifier] Prediction error: {e}")
            return {}

    def predict_batch(self, images: list[Image.Image], text: list[str]) -> list[dict[str, float]]:
        """
        배치 이미지에 대해 {텍스트: 확률} 딕셔너리 리스트 반환
        GPU 메모리 효율성을 위해 배치 처리
        """
        if not images:
            return []

        try:
            inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # [batch_size, num_prompts]
                probs = logits_per_image.softmax(dim=1)  # [batch_size, num_prompts]

            # 각 이미지에 대한 결과를 딕셔너리 리스트로 변환
            results = []
            for i in range(len(images)):
                result = {text[j]: probs[i][j].item() for j in range(len(text))}
                results.append(result)
            return results
        except Exception as e:
            print(f"[Classifier] Batch prediction error: {e}")
            return [{} for _ in images]

    def run(self, images: list[ImageItem]) -> list[ImageItem]:
        """
        FilterStep에서 상속됨.
        참고: 이 기본 run 메서드는 '무엇을 확인할지'에 대한 특정 로직이 필요할 수 있음.
        현재로서는 'keep_positive' 로직이 필요하다고 가정하지만, 'keep_positive'는 대상 키워드가 필요함.
        'keep_positive'가 명시적으로 호출되거나 기본 동작을 구성하는 것에 의존함.
        """
        # 파이프라인 단계와의 호환성을 위해 image.keyword가 사용 가능한 경우 기본값으로 사용
        return self.keep_positive(images)

    def keep_positive(self, images: list[ImageItem]) -> list[ImageItem]:
        kept_images = []
        rejected_count = 0

        print(f"[Classifier] Running existence check on {len(images)} images with batch processing...")
        total = len(images)
        batch_size = self.config.get("batch_size", 32)  # T4 GPU에 최적화된 배치 크기

        # 키워드별로 이미지 그룹화 (문제 1 해결: 다른 키워드 혼재 방지)
        from collections import defaultdict
        keyword_groups = defaultdict(list)

        for img_item in images:
            if not img_item.path:
                rejected_count += 1
                continue

            target_keyword = img_item.keyword
            if not target_keyword:
                target_keyword = self.config.get("target_class", "object")

            keyword_groups[target_keyword].append(img_item)

        processed_count = 0
        total_batches = sum((len(group) + batch_size - 1) // batch_size for group in keyword_groups.values())
        current_batch = 0

        # 키워드별로 배치 처리
        for keyword, group_images in keyword_groups.items():
            prompts = [
                f"a photo of {keyword}",
                "a photo of nothing",
                "text only",
                "random noise"
            ]

            # 배치 단위로 처리
            for batch_start in range(0, len(group_images), batch_size):
                batch_end = min(batch_start + batch_size, len(group_images))
                batch_items = group_images[batch_start:batch_end]
                current_batch += 1

                # 배치 준비: 이미지 로드 (문제 2 해결: context manager로 메모리 관리)
                batch_data = []  # (img_item, pil_img) 튜플 리스트

                for img_item in batch_items:
                    try:
                        pil_img = Image.open(img_item.path)
                        batch_data.append((img_item, pil_img))
                    except (UnidentifiedImageError, OSError) as e:
                        print(f"\n[Classifier] Invalid image file {img_item.path}: {e}")
                        rejected_count += 1
                    except Exception as e:
                        print(f"\n[Classifier] Error loading {img_item.path}: {e}")
                        rejected_count += 1

                if not batch_data:
                    processed_count += len(batch_items)
                    continue

                try:
                    # 배치 추론 실행
                    batch_images = [pil_img for _, pil_img in batch_data]
                    batch_scores = self.predict_batch(batch_images, prompts)

                    # 결과 처리
                    for (img_item, _), scores in zip(batch_data, batch_scores):
                        positive_score = scores.get(f"a photo of {keyword}", 0.0)
                        max_label = max(scores, key=scores.get) if scores else ""

                        if max_label == f"a photo of {keyword}" and positive_score > self.threshold:
                            img_item.meta["clip_check_score"] = positive_score
                            kept_images.append(img_item)
                        else:
                            rejected_count += 1

                except Exception as e:
                    print(f"\n[Classifier] Batch processing error: {e}")
                    # 배치 실패 시 개별 처리 (문제 8 해결: 에러 복구)
                    for img_item, pil_img in batch_data:
                        try:
                            scores = self.predict(pil_img, prompts)
                            positive_score = scores.get(f"a photo of {keyword}", 0.0)
                            max_label = max(scores, key=scores.get) if scores else ""

                            if max_label == f"a photo of {keyword}" and positive_score > self.threshold:
                                img_item.meta["clip_check_score"] = positive_score
                                kept_images.append(img_item)
                            else:
                                rejected_count += 1
                        except Exception as e2:
                            print(f"\n[Classifier] Individual processing failed {img_item.path}: {e2}")
                            rejected_count += 1
                finally:
                    # PIL 이미지 닫기 (문제 2 해결: 항상 정리)
                    for _, pil_img in batch_data:
                        try:
                            pil_img.close()
                        except:
                            pass

                processed_count += len(batch_items)

                # 진행상황 출력 (문제 9 해결: 정확한 진행률)
                print(f"[Classifier] Processed {processed_count}/{total} (kept: {len(kept_images)}, rejected: {rejected_count}, batch: {current_batch}/{total_batches})...", end="\r", flush=True)

        # GPU 메모리 정리 (문제 3 해결: 마지막에만 정리)
        if self.device == "cuda":
            torch.cuda.empty_cache()

        print()
        print(f"[Classifier] Kept {len(kept_images)} positive images. Rejected {rejected_count}.")
        return kept_images
