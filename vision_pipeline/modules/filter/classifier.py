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
        self.device = config.get("device", "cpu")
        
        if self.device == "mps" and not torch.backends.mps.is_available():
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
        {텍스트: 확률} 딕셔너리를 반환
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
        
        print(f"[Classifier] Running existence check on {len(images)} images...")
        
        for img_item in images:
            if not img_item.path:
                continue
                
            target_keyword = img_item.keyword
            if not target_keyword:
                 # 이미지에 키워드가 없는 경우, 설정 기본값으로 폴백하거나 건너뜀?
                 # 이상적으로는 설정의 'source' 또는 특정 'target_class'
                 target_keyword = self.config.get("target_class", "object")

            # [target_keyword, "not " + target_keyword] 또는 유사한 쌍을 비교?
            # 아니면 prob(target_keyword) > threshold만 확인?
            # CLIP은 비교와 함께 가장 잘 작동함.
            # ["a photo of {keyword}", "an image of text", "low quality image", "noise"] 사용
            # 또는 더 간단하게: ["a photo of {keyword}", "not {keyword}"]

            prompts = [
                f"a photo of {target_keyword}",
                "a photo of nothing",
                "text only",
                "random noise"
            ]

            try:
                with Image.open(img_item.path) as pil_img:
                    scores = self.predict(pil_img, prompts)

                # 긍정 프롬프트가 가장 높은지 또는 임계값 이상인지 확인
                positive_score = scores.get(f"a photo of {target_keyword}", 0.0)

                # 로직: 긍정 프롬프트가 최대 점수인가?
                max_label = max(scores, key=scores.get)

                if max_label == f"a photo of {target_keyword}" and positive_score > self.threshold:
                    img_item.meta["clip_check_score"] = positive_score
                    kept_images.append(img_item)
                else:
                    # print(f"Rejected {img_item.id}: Max={max_label} ({scores[max_label]:.2f}), Target={positive_score:.2f}")
                    rejected_count += 1
                    
            except (UnidentifiedImageError, OSError) as e:
                print(f"[Classifier] Invalid image file {img_item.path}: {e}")
                rejected_count += 1
            except Exception as e:
                print(f"[Classifier] Error processing {img_item.id}: {e}")
                rejected_count += 1

        print(f"[Classifier] Kept {len(kept_images)} positive images. Rejected {rejected_count}.")
        return kept_images
