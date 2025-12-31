import os
import base64
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from domain.label import LabelResult

# Load environment variables
load_dotenv()

class LLMVerifier:
    def __init__(self, config: dict):
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("[LLMVerifier] Warning: OPENAI_API_KEY not found in env. LLM verification will be skipped.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            
        self.model = config.get("model", "gpt-4o")
        self.system_prompt = config.get("system_prompt", "You are a helpful assistant.")
        self.max_tokens = config.get("max_tokens", 300)
        self.temperature = config.get("temperature", 0.0)

    def _encode_image(self, image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

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
                                    "url": f"data:image/jpeg;base64,{base64_image}"
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
                reason=f"Error: {e}"
            )
