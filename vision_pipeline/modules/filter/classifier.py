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
        Returns a dictionary of {text: probability}
        """
        try:
            inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)  # we can use softmax if we want probabilities across the provided classes

            # Convert result to dict
            result = {text[i]: probs[0][i].item() for i in range(len(text))}
            return result
        except Exception as e:
            print(f"[Classifier] Prediction error: {e}")
            return {}

    def run(self, images: list[ImageItem]) -> list[ImageItem]:
        """
        Inherited from FilterStep. 
        Note: This default run method might need specific logic for 'what to check against'.
        For now, it assumes 'keep_positive' logic is desired, but 'keep_positive' needs target keywords.
        We will rely on 'keep_positive' being called explicitly or configure a default behavior.
        """
        # For compatibility with Pipeline step, we can default to using image.keyword if available
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
                 # If no keyword on image, maybe fallback to config default or skip?
                 # ideally 'source' or a specific 'target_class' in config
                 target_keyword = self.config.get("target_class", "object")
            
            # We compare [target_keyword, "not " + target_keyword] or similar pairs?
            # Or just check if prob(target_keyword) > threshold?
            # CLIP works best with comparison. 
            # Let's use ["a photo of {keyword}", "an image of text", "low quality image", "noise"]
            # Or simpler: ["a photo of {keyword}", "not {keyword}"]
            
            prompts = [
                f"a photo of {target_keyword}", 
                "a photo of nothing", 
                "text only", 
                "random noise"
            ]
            
            try:
                with Image.open(img_item.path) as pil_img:
                    scores = self.predict(pil_img, prompts)
                
                # Check if the positive prompt is the highest or above threshold
                positive_score = scores.get(f"a photo of {target_keyword}", 0.0)
                
                # Logic: Is the positive prompt the max score?
                max_label = max(scores, key=scores.get)
                
                if max_label == f"a photo of {target_keyword}" and positive_score > self.threshold:
                    img_item.meta["param_check_score"] = positive_score
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
