"""Align VeRL rows and compute terminal-outcome GRPO for split trajectories."""

import numpy as np
import torch
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def compute_trajectory_grpo(batch, *, norm_adv_by_std):
    metadata = batch.non_tensor_batch
    final_rows = np.flatnonzero(metadata["rollout_final"])
    final_ids = metadata["rollout_trajectory"][final_rows]
    trajectories = metadata["rollout_trajectory"]
    row_by_trajectory = {trajectory: i for i, trajectory in enumerate(final_ids)}
    if len(row_by_trajectory) != len(final_ids) or set(row_by_trajectory) != set(trajectories):
        raise ValueError("Each sampled trajectory must have exactly one final fragment")
    fragment_to_outcome = [row_by_trajectory[trajectory] for trajectory in trajectories]
    if not np.array_equal(metadata["uid"], metadata["uid"][final_rows][fragment_to_outcome]):
        raise ValueError("All fragments of a trajectory must belong to the same input group")

    rewards = batch.batch["token_level_rewards"]
    outcomes = rewards[final_rows].sum(dim=-1, keepdim=True)
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=outcomes,
        response_mask=torch.ones_like(outcomes),
        index=metadata["uid"][final_rows],
        norm_adv_by_std_in_grpo=norm_adv_by_std,
    )
    response_mask = batch.batch["response_mask"]
    batch.batch["advantages"] = advantages[fragment_to_outcome] * response_mask
    batch.batch["returns"] = returns[fragment_to_outcome] * response_mask
    return batch
