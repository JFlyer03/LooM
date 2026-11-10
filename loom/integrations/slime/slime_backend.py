"""Token-preserving SGLang requests through slime's native router/client."""

from ..base import AsyncInferBackend


class SlimeBackend(AsyncInferBackend):
    def __init__(self, infer_engine, tokenizer=None, cfg=None):
        self.client = infer_engine

    async def async_generate_prompts(self, prompts, sampling_params, **kwargs):
        raise NotImplementedError("Memory rollout uses token inputs")

    async def async_generate_ids(self, input_ids, sampling_params, **kwargs):
        params = dict(sampling_params)
        params.pop("n", None)
        params["max_new_tokens"] = params.pop("max_tokens")
        output = await self.client.generate(input_ids, params, **kwargs)
        meta = output["meta_info"]
        logprobs = meta.get("output_token_logprobs")
        if logprobs is None:
            raise ValueError("SGLang must return output_token_logprobs for memory training")
        finish = meta["finish_reason"]
        if isinstance(finish, dict):
            finish = finish["type"]
        if finish not in ("stop", "length"):
            raise RuntimeError(f"SGLang request did not finish successfully: {finish}")
        return output["text"], {
            "output_tokens": [item[1] for item in logprobs],
            "logprobs": [item[0] for item in logprobs],
            "finish_reason": finish,
        }
