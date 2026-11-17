"""Validate the native memory training configuration before Ray startup."""

import torch


def validate_memory_training(config):
    """Fail before starting Ray when this MemAgent path cannot execute correctly."""
    from omegaconf import OmegaConf

    task_path = OmegaConf.select(config, "skyrl_agent.task_yaml")
    if not task_path:
        return
    task = OmegaConf.load(task_path)
    if task.agent_cls != "loom.agents.memory.MemAgent":
        return
    if task.generator.infer_backend != "verl":
        raise ValueError("MemAgent's VeRL task must set generator.infer_backend=verl")
    if config.algorithm.adv_estimator != "grpo" or config.algorithm.use_kl_in_reward:
        raise ValueError("MemAgent's VeRL adapter requires GRPO and use_kl_in_reward=false; use actor KL loss")
    rollout = config.actor_rollout_ref.rollout
    if rollout.name != "vllm" or rollout.mode != "async":
        raise ValueError("MemAgent's VeRL adapter requires asynchronous vLLM rollout (mode=async)")
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError("MemAgent's asynchronous vLLM rollout requires CUDA compute capability >= 8.0")
