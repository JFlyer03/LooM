"""Map MemAgent's task settings to VeRL's native Hydra overrides."""

import json
from pathlib import Path

from omegaconf import OmegaConf


DEFAULT_TASK = str(Path(__file__).resolve().parents[3] / "examples/MemAgent/memagent_verl.yaml")


def prepare_memory_overrides(args):
    overrides = OmegaConf.from_dotlist([
        arg.lstrip("+") for arg in args if "=" in arg and not arg.startswith("--")
    ])
    task_path = OmegaConf.select(overrides, "skyrl_agent.task_yaml", default=DEFAULT_TASK)
    task = OmegaConf.load(task_path)
    if task.agent_cls != "loom.agents.memory.MemAgent" or task.generator.infer_backend != "verl":
        raise ValueError("The VeRL MemAgent entrypoint requires a MemAgent task with infer_backend: verl")
    gen = task.generator
    output_length = max(task.memory.max_memorization_length, task.memory.max_final_response_length)
    if any(type(v) is not int or v < 1 for v in (
        gen.num_trajectories, gen.val_config.num_trajectories, gen.max_prompt_length, output_length,
    )) or gen.num_trajectories < 2:
        raise ValueError("Token limits must be positive integers and GRPO needs at least two trajectories")
    shared = {
        "skyrl_agent.task_yaml": str(task_path),
        "skyrl_agent.num_trajectories": gen.num_trajectories,
        "data.max_prompt_length": gen.max_prompt_length,
        "data.max_response_length": output_length,
        "actor_rollout_ref.rollout.prompt_length": gen.max_prompt_length,
        "actor_rollout_ref.rollout.response_length": output_length,
        "actor_rollout_ref.rollout.max_model_len": gen.max_prompt_length + output_length,
        "actor_rollout_ref.rollout.n": gen.num_trajectories,
        "actor_rollout_ref.rollout.temperature": gen.sampling_params.temperature,
        "actor_rollout_ref.rollout.top_p": gen.sampling_params.top_p,
        "actor_rollout_ref.rollout.val_kwargs.n": gen.val_config.num_trajectories,
        "actor_rollout_ref.rollout.val_kwargs.temperature": gen.val_config.sampling_params.temperature,
        "actor_rollout_ref.rollout.val_kwargs.top_p": gen.val_config.sampling_params.top_p,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": gen.val_config.sampling_params.temperature > 0,
    }
    if OmegaConf.select(overrides, "actor_rollout_ref.rollout.mode") == "sync":
        shared["actor_rollout_ref.rollout.memory_task_yaml"] = str(task_path)
    additions = []
    for key, value in shared.items():
        if OmegaConf.select(overrides, key, default=value) != value:
            raise ValueError(f"{key} must match {task_path} ({value!r}); change the task YAML instead")
        if OmegaConf.select(overrides, key) is None:
            prefix = "+" if key.startswith("skyrl_agent.") or key.endswith(".memory_task_yaml") else ""
            additions.append(f"{prefix}{key}={json.dumps(value)}")
    return [*additions, *args]
