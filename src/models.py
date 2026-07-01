import os
import json
import re
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from huggingface_hub import get_token

logger = logging.getLogger(__name__)

_HF_TOKEN = get_token()


def parse_model_name(model_name: str) -> tuple[str, str]:
    if model_name.startswith("openai/"):
        return "openai", model_name.removeprefix("openai/")
    if model_name.startswith("google/"):
        return "google", model_name.removeprefix("google/")
    if model_name.startswith("xai/"):
        return "xai", model_name.removeprefix("xai/")
    if model_name.startswith("hf-api/"):
        return "hf-api", model_name.removeprefix("hf-api/")
    return "hf", model_name


def create_client(model_name: str, hf_cache_dir: str = "models") -> "BaseClient":
    provider, actual_name = parse_model_name(model_name)
    if provider == "openai":
        return OpenAIClient(actual_name)
    elif provider == "google":
        return GeminiClient(actual_name)
    elif provider == "xai":
        return XAIClient(actual_name)
    elif provider == "hf-api":
        return HFInferenceAPIClient(actual_name)
    else:
        return HFClient(actual_name, hf_cache_dir)


class BaseClient(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.01, max_tokens: int = 4096
    ) -> str: ...

    def generate_structured(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.01, max_retries: int = 3
    ) -> dict[str, Any] | None:
        from .utils import extract_json_from_response

        for attempt in range(max_retries):
            raw = self.generate(system_prompt, user_prompt, temperature=temperature)
            parsed = extract_json_from_response(raw)
            if parsed is not None:
                return parsed
            if attempt < max_retries - 1:
                user_prompt += (
                    "\n\nYour previous response was not valid JSON. "
                    "Respond with ONLY valid JSON matching the schema. "
                    "No explanation, no markdown formatting."
                )
        logger.error(f"Failed to get valid JSON after {max_retries} attempts from {self.model_name}")
        return None


class HFClient(BaseClient):
    def __init__(self, model_name: str, cache_dir: str = "models"):
        super().__init__(model_name)
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from huggingface_hub.utils import HfHubHTTPError

        cache_path = Path(self.cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        logger.info(f"Loading {self.model_name} on local GPU (4-bit)...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(cache_path),
                trust_remote_code=True,
                token=_HF_TOKEN,
            )
        except (OSError, HfHubHTTPError) as e:
            msg = str(e)
            if "gated" in msg or "403" in msg:
                print()
                print(f"❌ Cannot access {self.model_name}.")
                print("   This model (Llama) requires you to accept the license on HuggingFace:")
                print(f"     https://huggingface.co/{self.model_name}")
                print()
                print("   Or use a non-gated model instead:")
                print("     prs evaluate -m mistralai/Mistral-7B-Instruct-v0.3")
                print()
            raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                cache_dir=str(cache_path),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                token=_HF_TOKEN,
            )
        except (OSError, HfHubHTTPError) as e:
            msg = str(e)
            if "gated" in msg or "403" in msg:
                print()
                print(f"❌ Cannot access {self.model_name}.")
                print("   This model (Llama) requires you to accept the license on HuggingFace:")
                print(f"     https://huggingface.co/{self.model_name}")
                print()
                print("   Or use a non-gated model instead:")
                print("     prs evaluate -m mistralai/Mistral-7B-Instruct-v0.3")
                print()
            raise

        import torch
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        logger.info(f"Running {self.model_name} locally on {device_name}")

    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.01, max_tokens: int = 4096
    ) -> str:
        self._load()
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()


class HFInferenceAPIClient(BaseClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        if not _HF_TOKEN:
            raise ValueError(
                "HF token not found. Run: huggingface-cli login"
            )

    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.01, max_tokens: int = 4096
    ) -> str:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=_HF_TOKEN)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"Querying HF Inference API: {self.model_name}")
        response = client.chat_completion(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


class OpenAIClient(BaseClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.01, max_tokens: int = 4096
    ) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content.strip()


class GeminiClient(BaseClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.01, max_tokens: int = 4096
    ) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        response = model.generate_content(user_prompt)
        return response.text.strip()


class XAIClient(BaseClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.api_key = os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY environment variable not set")

    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.01, max_tokens: int = 4096
    ) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
