"""Pack LooM rollout rows for VeRL's synchronous and asynchronous workers."""

import numpy as np
import torch
from tensordict import TensorDict
from verl.protocol import DataProto


def pack_rollout(inputs, tokenizer, rollout_config, val_mode=False):
    memory_rows = "trajectory_keys" in inputs
    if memory_rows and val_mode:
        # Validation reports complete trajectories, so keep only final answers.
        keep = [i for i, last in enumerate(inputs["is_last_episode"]) if last]
        inputs = {**inputs, **{
            key: [inputs[key][i] for i in keep] for key in (
                "prompt_token_ids", "response_ids", "loss_masks", "rewards",
                "trajectory_keys", "is_last_episode",
            )
        }}
    # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
    # prompts: left pad
    # responses: right pad
    # input_ids: prompt + response
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

    # inputs
    tokenizer.padding_side = "left"
    max_prompt_length = max(
        max([len(input_ids) for input_ids in inputs["prompt_token_ids"]]),
        rollout_config.prompt_length,
    )
    outputs = tokenizer.pad(
        [{"input_ids": input_ids} for input_ids in inputs["prompt_token_ids"]],
        padding="max_length",
        max_length=max_prompt_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

    # responses
    tokenizer.padding_side = "right"
    max_response_length = max(
        max([len(response) for response in inputs["response_ids"]]),
        rollout_config.response_length,
    )
    outputs = tokenizer.pad(
        [{"input_ids": response_ids} for response_ids in inputs["response_ids"]],
        padding="max_length",
        max_length=max_response_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

    # response_mask
    response_length = response_ids.shape[1]
    loss_masks = [loss_mask + [0] * (response_length - len(loss_mask)) for loss_mask in inputs["loss_masks"]]
    response_mask = torch.tensor(loss_masks, dtype=torch.long)
    assert (
        response_ids.shape == response_mask.shape
    ), f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
    response_mask = response_mask * response_attention_mask

    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

    batch = TensorDict(
        {
            "prompts": prompt_ids,  # [bsz, prompt_length]
            "responses": response_ids,  # [bsz, response_length]
            "response_mask": response_mask,  # [bsz, response_length]
            "input_ids": input_ids,  # [bsz, prompt_length + response_length]
            "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
            "position_ids": position_ids,  # [bsz, prompt_length + response_length]
        },
        batch_size=len(input_ids),
    )

    non_tensor_batch = {"rewards": np.array(inputs["rewards"])}
    if memory_rows:
        non_tensor_batch.update({
            "_loom_source_index": np.array([int(key[0]) for key in inputs["trajectory_keys"]]),
            "_loom_repetition_id": np.array([key[1] for key in inputs["trajectory_keys"]]),
            "_loom_is_last_step": np.array(inputs["is_last_episode"], dtype=bool),
        })
    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info={"rollout_metrics": inputs["rollout_metrics"], "timing": {}},
    )

