import os
import io
import json
import base64
from typing import Optional, Tuple

from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert image labeler. "
    "Return a concise, specific class label for the main object in the image. "
    "Use 1-3 words in lowercase or snake_case. "
    "If the image is unclear, return 'unknown'."
)

class VLMLabeler:
    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}
        self.config = config

        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("[VLMLabeler] Warning: OPENAI_API_KEY not found. VLM labeling will be skipped.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

        self.model = config.get("model", "gpt-4o-mini")
        self.system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.max_tokens = config.get("max_tokens", 120)
        self.temperature = config.get("temperature", 0.0)
        self.default_label = config.get("default_label", "unknown")

    def is_available(self) -> bool:
        return self.client is not None

    def _encode_pil_image(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def label_image(self, image: Image.Image) -> Tuple[str, Optional[float]]:
        if not self.client:
            return self.default_label, 0.0

        try:
            base64_image = self._encode_pil_image(image)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Return ONLY valid JSON with this schema:\n"
                                    '{ "label": string, "confidence": number }\n'
                                    "The label must be a concise class name."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    },
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result_json = json.loads(content)

            label = result_json.get("label") or self.default_label
            confidence = result_json.get("confidence", None)
            return str(label).strip(), confidence
        except Exception as e:
            print(f"[VLMLabeler] Error labeling image: {e}")
            return self.default_label, 0.0
