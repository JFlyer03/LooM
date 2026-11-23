from typing import Any, List, NotRequired, TypedDict
from loom.integrations.base import (
    AsyncInferBackend,
    GeneratorOutput,
    GeneratorInput,
    register_backend,
    BackendSpec,
)
import os
import aiohttp


class OpenAIBackendConfig(TypedDict):
    model_name: str
    api_url: str
    model_max_len: NotRequired[int]


class OpenAIBackend(AsyncInferBackend):
    def __init__(self, infer_engine: Any, cfg: OpenAIBackendConfig):
        assert os.environ.get("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"
        self.model_name = cfg["model_name"]
        self.api_url = cfg["api_url"]
        self.model_max_len = cfg.get("model_max_len")

    async def async_generate_prompts(self, prompts: str, sampling_params: dict, **kwargs) -> str:
        base_url = self.api_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        payload = dict(sampling_params)
        payload["model"] = self.model_name
        payload["messages"] = [{"role": "user", "content": prompts}]
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as response:
                response.raise_for_status()
                output = await response.json()
        try:
            text = output["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Chat completions response is missing generated text") from exc
        if not isinstance(text, str):
            raise ValueError("Chat completions response is missing generated text")
        return text

    async def async_generate_ids(self, input_ids: List[int], sampling_params: dict, **kwargs) -> str:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            headers = {"Content-Type": "application/json"}
            if not isinstance(sampling_params, dict):
                sampling_params = dict(sampling_params)
            payload = sampling_params.copy()
            payload["model"] = self.model_name
            # payload["max_tokens"] = self.model_max_len - len(input_ids) - 1
            print(f"max tokens: {payload['max_tokens']}")

            payload["prompt"] = input_ids
            output = await session.post(f"{self.api_url}/v1/completions", json=payload, headers=headers)
            output = await output.json()

        meta_info = {
            "output_tokens": None,
            "finish_reason": output["choices"][0]["finish_reason"],
            "logprobs": None,
        }

        return output["choices"][0]["text"], meta_info


class OpenAIGeneratorOutput(GeneratorOutput):
    def __init__(self, result: Any):
        self.result = result


class OpenAIGeneratorInput(GeneratorInput):
    def __init__(self, input_batch: Any):
        self.input_batch = input_batch


register_backend(
    "openai_server",
    BackendSpec(
        infer_backend_cls=OpenAIBackend,
        generator_output_cls=OpenAIGeneratorOutput,
        generator_input_cls=OpenAIGeneratorInput,
    ),
)
