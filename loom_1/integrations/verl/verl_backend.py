from loom)1.integrations.base import AsyncInferBackend, GeneratorOutput, GeneratorInput
from typing import Any, List, Dict
from loguru import logger


class VeRLBackend(AsyncInferBackend):
    def __init__(self, infer_engine, tokenizer: Any = None, cfg: Dict[str, Any] = None):
        self.infer_engine = infer_engine
        self.tokenizer = tokenizer

    async def async_generate_ids(
        self,
        input_ids: List[int],
        sampling_params: Dict[str, Any],
        request_id: str,
        **kwargs,
    ):
        output = await self.infer_engine.generate(
            request_id=request_id,
            prompt_ids=input_ids,
            sampling_params=sampling_params,
        )
        # Current VERL's LLMServerClient returns TokenOutput, not a text tuple.
        if hasattr(output, "token_ids"):
            tokens = list(output.token_ids)
            return self.tokenizer.decode(tokens, skip_special_tokens=True), {
                "output_tokens": tokens,
                "finish_reason": "stop" if output.stop_reason == "completed" else output.stop_reason,
                "logprobs": output.log_probs,
            }
        return output

    async def async_generate_prompts(self, prompts: Any, sampling_params: Any) -> List[str]:
        raise NotImplementedError


class VeRLGeneratorOutput(GeneratorOutput):
    def __init__(self, result: Dict[str, Any]):
        self.result = result


class VeRLGeneratorInput(GeneratorInput):
    def __init__(self, input_batch: Any):
        if isinstance(input_batch, list):
            self.input_batch = input_batch
            return
        self.input_batch: List[Dict[str, Any]] = []
        non_tensor_batch = input_batch.non_tensor_batch
        first_key = next(iter(non_tensor_batch.keys()))
        num_entries = len(non_tensor_batch[first_key])
        for i in range(num_entries):
            data_item: dict = {key: non_tensor_batch[key][i] for key in non_tensor_batch.keys()}
            data_item["__loom_uid"] = str(i)
            self.input_batch.append(data_item)
        logger.info(f"input batch of size: {len(self.input_batch)}")
        logger.info(f"keys: {self.input_batch[0].keys()}")
