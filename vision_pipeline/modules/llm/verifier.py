import os
import base64
import json
import asyncio
import io
from pathlib import Path
from PIL import Image
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from domain.label import LabelResult
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

class LLMVerifier:
    def __init__(self, config: dict):
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("[LLMVerifier] Warning: OPENAI_API_KEY not found in env. LLM verification will be skipped.")
            self.client = None
            self.async_client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)

        self.model = config.get("model", "gpt-4o-mini")
        self.system_prompt = config.get("system_prompt", "You are a helpful assistant.")
        self.max_tokens = config.get("max_tokens", 300)
        self.temperature = config.get("temperature", 0.0)

    def _encode_image(self, image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _encode_pil_image(self, image: Image.Image) -> str:
        """PIL 이미지를 base64로 인코딩"""
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def verify_image(self, image_path: str | Path, label: str) -> LabelResult:
        if not self.client:
             return LabelResult(image_id="unknown", crop_path=str(image_path), verified=False, label=label, reason="No API Key/Client", confidence=0.0)

        try:
            base64_image = self._encode_image(str(image_path))

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Is this a {label}? Respond in JSON."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # 85 tokens, $0.0002 per image (gpt-4o)
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={ "type": "json_object" }
            )

            content = response.choices[0].message.content
            result_json = json.loads(content)

            return LabelResult(
                image_id="", # filled by caller
                crop_path=str(image_path),
                verified=result_json.get("verified", False),
                label=label,
                reason=result_json.get("reason", "No reason provided"),
                confidence=result_json.get("confidence", 0.0)
            )

        except Exception as e:
            print(f"[LLMVerifier] Error verifying {image_path}: {e}")
            return LabelResult(
                image_id="",
                crop_path=str(image_path),
                verified=False,
                label=label,
                reason=f"Error: {e}",
                confidence=0.0
            )

    def verify_pil_image(self, image: Image.Image, label: str) -> LabelResult:
        """PIL 이미지를 검증"""
        if not self.client:
            return LabelResult(image_id="", crop_path="", verified=False, label=label, reason="No API Key/Client", confidence=0.0)

        try:
            base64_image = self._encode_pil_image(image)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Is this a {label}? Respond in JSON."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # 85 tokens, $0.0002 per image (gpt-4o)
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={ "type": "json_object" }
            )

            content = response.choices[0].message.content
            result_json = json.loads(content)

            return LabelResult(
                image_id="",
                crop_path="",
                verified=result_json.get("verified", False),
                label=label,
                reason=result_json.get("reason", "No reason provided"),
                confidence=result_json.get("confidence", 0.0)
            )

        except Exception as e:
            print(f"[LLMVerifier] Error verifying PIL image: {e}")
            return LabelResult(
                image_id="",
                crop_path="",
                verified=False,
                label=label,
                reason=f"Error: {e}",
                confidence=0.0
            )

    async def verify_image_async(self, image_path: str | Path, label: str) -> LabelResult:
        """비동기 이미지 검증"""
        if not self.async_client:
            return LabelResult(image_id="unknown", crop_path=str(image_path), verified=False, label=label, reason="No API Key/Client", confidence=0.0)

        try:
            base64_image = self._encode_image(str(image_path))

            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Is this a {label}? Respond in JSON."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # 85 tokens, $0.0002 per image (gpt-4o)
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={ "type": "json_object" }
            )

            content = response.choices[0].message.content
            result_json = json.loads(content)

            return LabelResult(
                image_id="",
                crop_path=str(image_path),
                verified=result_json.get("verified", False),
                label=label,
                reason=result_json.get("reason", "No reason provided"),
                confidence=result_json.get("confidence", 0.0)
            )

        except Exception as e:
            print(f"[LLMVerifier] Async error verifying {image_path}: {e}")
            return LabelResult(
                image_id="",
                crop_path=str(image_path),
                verified=False,
                label=label,
                reason=f"Error: {e}",
                confidence=0.0
            )

    async def verify_batch_async(self, crop_label_pairs: list[tuple[str, str]]) -> list[LabelResult]:
        """배치 비동기 검증 (crop_path, label) 튜플 리스트"""
        tasks = [self.verify_image_async(crop_path, label) for crop_path, label in crop_label_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                crop_path, label = crop_label_pairs[i]
                final_results.append(LabelResult(
                    image_id="",
                    crop_path=str(crop_path),
                    verified=False,
                    label=label,
                    reason=f"Exception: {result}",
                    confidence=0.0
                ))
            else:
                final_results.append(result)

        return final_results
