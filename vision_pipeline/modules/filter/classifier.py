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

        # 배치 단위로 처리
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_items = images[batch_start:batch_end]

            # 배치 준비: 이미지 로드 및 프롬프트 생성
            batch_images = []
            batch_indices = []
            batch_prompts_list = []

            for idx, img_item in enumerate(batch_items):
                if not img_item.path:
                    continue

                target_keyword = img_item.keyword
                if not target_keyword:
                    target_keyword = self.config.get("target_class", "object")

                try:
                    pil_img = Image.open(img_item.path)
                    batch_images.append(pil_img)
                    batch_indices.append(idx)
                    batch_prompts_list.append((
                        target_keyword,
                        [
                            f"a photo of {target_keyword}",
                            "a photo of nothing",
                            "text only",
                            "random noise"
                        ]
                    ))
                except (UnidentifiedImageError, OSError) as e:
                    print(f"\n[Classifier] Invalid image file {img_item.path}: {e}")
                    rejected_count += 1
                except Exception as e:
                    print(f"\n[Classifier] Error loading {img_item.path}: {e}")
                    rejected_count += 1

            if not batch_images:
                continue

            # 모든 이미지가 같은 키워드를 가진다고 가정 (첫 번째 프롬프트 사용)
            # 만약 다른 키워드가 필요하면 개별 처리 필요
            prompts = batch_prompts_list[0][1]

            try:
                # 배치 추론 실행
                batch_scores = self.predict_batch(batch_images, prompts)

                # 결과 처리
                for idx, scores in zip(batch_indices, batch_scores):
                    img_item = batch_items[idx]
                    target_keyword = batch_prompts_list[batch_indices.index(idx)][0]

                    positive_score = scores.get(f"a photo of {target_keyword}", 0.0)
                    max_label = max(scores, key=scores.get) if scores else ""

                    if max_label == f"a photo of {target_keyword}" and positive_score > self.threshold:
                        img_item.meta["clip_check_score"] = positive_score
                        kept_images.append(img_item)
                    else:
                        rejected_count += 1

                # GPU 메모리 정리
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n[Classifier] Batch processing error: {e}")
                # 배치 실패 시 모두 거부
                rejected_count += len(batch_images)
            finally:
                # PIL 이미지 닫기
                for img in batch_images:
                    img.close()

            # 진행상황 출력
            processed = min(batch_end, total)
            print(f"[Classifier] Processed {processed}/{total} (kept: {len(kept_images)}, rejected: {rejected_count})...", end="\r", flush=True)

        print()
        print(f"[Classifier] Kept {len(kept_images)} positive images. Rejected {rejected_count}.")
        return kept_images
