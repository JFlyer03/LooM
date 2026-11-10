"""slime custom-generate hook: one execution produces independently trained steps."""

import os
from collections import defaultdict, deque
from copy import deepcopy

from ..memory_episode import run_memory_episode


class RouterClient:
    def __init__(self, args, sample):
        self.args = args
        self.sample = sample
        self.request_versions = defaultdict(deque)

    async def generate(self, input_ids, sampling_params, **kwargs):
        from slime.rollout.sglang_rollout import get_model_url
        from slime.utils.http_utils import post

        headers = None
        if getattr(self.args, "router_policy", None) == "consistent_hashing":
            headers = {"X-SMG-Routing-Key": str(self.sample.session_id)}
        output = await post(
            get_model_url(self.args, getattr(self.args, "actor_model_name", "actor")),
            {"input_ids": input_ids, "sampling_params": sampling_params, "return_logprob": True},
            headers=headers,
        )
        version = output["meta_info"].get("weight_version")
        tokens = tuple(item[1] for item in output["meta_info"]["output_token_logprobs"])
        self.request_versions[(tuple(input_ids), tokens)].append(version)
        return output


async def generate(args, sample, sampling_params, evaluation=False):
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.types import Sample

    task_yaml = getattr(args, "loom_task", None) or os.environ.get("LOOM_SLIME_TASK")
    if not task_yaml:
        raise ValueError("Set LOOM_SLIME_TASK to the slime memory task YAML")
    instance = sample.metadata.get("instance")
    if instance is None:
        raise ValueError("slime samples need metadata.instance containing the complete memory task")
    client = RouterClient(args, sample)
    params = dict(sampling_params)
    if "max_new_tokens" in params:
        params["max_tokens"] = params.pop("max_new_tokens")
    result = await run_memory_episode(
        task_yaml, client, GenerateState(args).tokenizer,
        {"instance": instance, "__loom_uid": str(sample.index), "data_source": "loom_memagent"},
        params, val_mode=evaluation,
    )
    segments = []
    for i, response in enumerate(result["response_ids"]):
        segment = deepcopy(sample)
        segment.rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
        segment.tokens = result["prompt_token_ids"][i] + response
        segment.response = GenerateState(args).tokenizer.decode(response, skip_special_tokens=True)
        segment.response_length = len(response)
        segment.loss_mask = result["loss_masks"][i]
        segment.rollout_log_probs = result["rollout_logprobs"][i]
        segment.reward = float(result["traj_rewards"][0])
        segment.status = Sample.Status.TRUNCATED if result["stop_reasons"][i] == "length" else Sample.Status.COMPLETED
        version = client.request_versions[(tuple(result["prompt_token_ids"][i]), tuple(response))].popleft()
        segment.weight_versions = [] if version is None else [str(version)]
        segment.metadata = {**segment.metadata, "loom_last_step": result["is_last_episode"][i]}
        segments.append(segment)
    # Native evaluation counts samples, so report each trajectory's final answer once.
    return segments[-1] if evaluation else segments
