import os
import json
import re
import time
import threading
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from huggingface_hub import get_token

logger = logging.getLogger(__name__)

_HF_TOKEN = get_token()

_rate_limiters: dict[str, "RateLimiter"] = {}
_rate_limiters_lock = threading.Lock()


class RateLimiter:
    def __init__(self, requests_per_minute: float):
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.lock = threading.Lock()
        self.last_request_time = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        with self.lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.info(f"Rate limiter: waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
            self.last_request_time = time.time()


def _get_rate_limiter(provider: str, rpm: float = 60) -> "RateLimiter":
    with _rate_limiters_lock:
        if provider not in _rate_limiters:
            _rate_limiters[provider] = RateLimiter(rpm)
        return _rate_limiters[provider]


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


def create_client(
    model_name: str,
    hf_cache_dir: str = "models",
    rate_limits: dict[str, int] | None = None,
) -> "BaseClient":
    rate_limits = rate_limits or {}
    provider, actual_name = parse_model_name(model_name)
    rpm = rate_limits.get(provider, 60)
    if provider == "openai":
        return OpenAIClient(actual_name, rpm)
    elif provider == "google":
        return GeminiClient(actual_name, rpm)
    elif provider == "xai":
        return XAIClient(actual_name, rpm)
    elif provider == "hf-api":
        return HFInferenceAPIClient(actual_name, rpm)
    else:
        return HFClient(actual_name, hf_cache_dir)


class BaseClient(ABC):
    def __init__(self, model_name: str, rpm: float = 60):
        self.model_name = model_name
        self.rate_limiter = _get_rate_limiter(self.provider_name(), rpm)

    @classmethod
    @abstractmethod
    def provider_name(cls) -> str: ...

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

    def _call_with_retry(self, fn, *args, max_retries: int = 5, **kwargs):
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                delay = self._parse_retry_delay(e)
                if delay is None:
                    raise
                logger.warning(
                    f"Rate limited ({type(e).__name__}), "
                    f"waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
        raise RuntimeError(f"API call failed after {max_retries} retries due to rate limiting")

    def _parse_retry_delay(self, error: Exception) -> float | None:
        return None


class HFClient(BaseClient):
    def __init__(self, model_name: str, cache_dir: str = "models"):
        super().__init__(model_name, rpm=0)
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None

    @classmethod
    def provider_name(cls) -> str:
        return "hf"

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
    def __init__(self, model_name: str, rpm: float = 30):
        super().__init__(model_name, rpm=rpm)
        if not _HF_TOKEN:
            raise ValueError(
                "HF token not found. Run: huggingface-cli login"
            )

    @classmethod
    def provider_name(cls) -> str:
        return "hf-api"

    def _parse_retry_delay(self, error: Exception) -> float | None:
        import requests
        if isinstance(error, requests.exceptions.HTTPError) and error.response.status_code == 429:
            retry_after = error.response.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after) + 1
                except ValueError:
                    pass
        return None

    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.01, max_tokens: int = 4096
    ) -> str:
        self.rate_limiter.wait()
        return self._call_with_retry(self._hfapi_generate, system_prompt, user_prompt, temperature, max_tokens)

    def _hfapi_generate(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
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
    def __init__(self, model_name: str, rpm: float = 60):
        super().__init__(model_name, rpm=rpm)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

    @classmethod
    def provider_name(cls) -> str:
        return "openai"

    def _parse_retry_delay(self, error: Exception) -> float | None:
        import openai
        if isinstance(error, openai.RateLimitError):
            try:
                return float(error.response.headers.get("retry-after-ms", 0)) / 1000 + 1
            except (TypeError, AttributeError):
                pass
        return None

    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.01, max_tokens: int = 4096
    ) -> str:
        self.rate_limiter.wait()
        return self._call_with_retry(self._openai_generate, system_prompt, user_prompt, temperature, max_tokens)

    def _openai_generate(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
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
    def __init__(self, model_name: str, rpm: float = 5):
        super().__init__(model_name, rpm=rpm)
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

    @classmethod
    def provider_name(cls) -> str:
        return "google"

    def _parse_retry_delay(self, error: Exception) -> float | None:
        msg = str(error)
        m = re.search(r"Please retry in\s+([\d.]+)s", msg)
        if m:
            return float(m.group(1)) + 1
        return None

    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.01, max_tokens: int = 4096
    ) -> str:
        self.rate_limiter.wait()
        return self._call_with_retry(self._gemini_generate, system_prompt, user_prompt, temperature, max_tokens)

    def _gemini_generate(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        return response.text.strip()


class XAIClient(BaseClient):
    def __init__(self, model_name: str, rpm: float = 60):
        super().__init__(model_name, rpm=rpm)
        self.api_key = os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY environment variable not set")

    @classmethod
    def provider_name(cls) -> str:
        return "xai"

    def _parse_retry_delay(self, error: Exception) -> float | None:
        import openai
        if isinstance(error, openai.RateLimitError):
            try:
                return float(error.response.headers.get("retry-after-ms", 0)) / 1000 + 1
            except (TypeError, AttributeError):
                pass
        return None

    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.01, max_tokens: int = 4096
    ) -> str:
        self.rate_limiter.wait()
        return self._call_with_retry(self._xai_generate, system_prompt, user_prompt, temperature, max_tokens)

    def _xai_generate(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
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
