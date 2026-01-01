import os
import io
import json
import base64
import re
import time
from typing import Optional, Tuple

from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

DEFAULT_SYSTEM_PROMPT = (
    "You are a final character selection judge. "
    "Choose exactly one label from the provided candidates. "
    "If uncertain, choose unknown. "
    "Return only valid JSON with a single key: label."
)

class VLMLabeler:
    """VLM(Vision Language Model)을 사용하여 이미지 라벨링을 수행하는 클래스"""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}
        self.config = config

        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("[VLMLabeler] 경고: OPENAI_API_KEY를 찾을 수 없습니다. VLM 라벨링을 건너뜁니다.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

        self.model = config.get("model", "gpt-4o-mini")
        self.system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.max_tokens = config.get("max_tokens", 120)
        self.temperature = config.get("temperature", 0.0)
        self.default_label = config.get("default_label", "unknown")
        self.rate_limit_max_retries = int(config.get("rate_limit_max_retries", 2))
        self.rate_limit_retry_default_delay = float(config.get("rate_limit_retry_default_delay", 1.0))

    def is_available(self) -> bool:
        """VLM 클라이언트가 사용 가능한지 확인"""
        return self.client is not None

    def _encode_pil_image(self, image: Image.Image) -> str:
        """PIL 이미지를 base64로 인코딩"""
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _get_rate_limit_delay(self, error: Exception) -> Optional[float]:
        """Rate limit 오류일 때 재시도 대기 시간을 추출"""
        message = str(error)
        if "rate_limit" not in message.lower() and "rate limit" not in message.lower() and "429" not in message:
            return None
        match = re.search(r"Please try again in (\d+)ms", message, re.IGNORECASE)
        if match:
            delay_ms = int(match.group(1))
            return max(delay_ms / 1000.0, 0.001)
        return self.rate_limit_retry_default_delay

    def label_image(self, image: Image.Image, candidates: Optional[list[str]] = None) -> Tuple[str, Optional[float]]:
        """이미지를 VLM으로 라벨링하여 라벨을 반환"""
        if not self.client:
            return self.default_label, 0.0
        if not candidates:
            return self.default_label, 0.0

        try:
            base64_image = self._encode_pil_image(image)
        except Exception as e:
            print(f"[VLMLabeler] 이미지 인코딩 오류: {e}")
            return self.default_label, 0.0

        candidate_list = [str(c).strip() for c in candidates if str(c).strip()]
        if self.default_label not in candidate_list:
            candidate_list.append(self.default_label)
        if not candidate_list:
            return self.default_label, 0.0

        payload = {
            "task": "final_character_selection",
            "rules": {"choose_one": True, "fallback": self.default_label},
            "candidates": candidate_list,
        }

        retries_left = max(self.rate_limit_max_retries, 0)
        while True:
            try:
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
                                        json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
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

                label = str(result_json.get("label") or self.default_label).strip()
                if label not in candidate_list:
                    label = self.default_label
                return label, None
            except Exception as e:
                retry_delay = self._get_rate_limit_delay(e)
                if retry_delay is not None and retries_left > 0:
                    print(f"[VLMLabeler] Rate limit hit. Retrying in {retry_delay:.3f}s...")
                    time.sleep(retry_delay)
                    retries_left -= 1
                    continue
                print(f"[VLMLabeler] 이미지 라벨링 오류: {e}")
                return self.default_label, 0.0
