"""Map the task's shared generation settings to native SkyRL configuration."""

from pathlib import Path

from omegaconf import OmegaConf


DEFAULT_TASK = str(Path(__file__).resolve().parents[3] / "examples/MemAgent/memagent.yaml")


def prepare_memory_config(args):
    if any(arg.startswith("+") for arg in args):
        raise ValueError("The '+' prefix is not supported by the native SkyRL entrypoint")
    overrides = OmegaConf.from_dotlist(args)
    task_path = OmegaConf.select(overrides, "generator.task", default=DEFAULT_TASK)
    task = OmegaConf.load(task_path)
    gen = task.generator
    max_output = max(
        task.memory.get("max_memorization_length", gen.sampling_params.max_tokens),
        task.memory.get("max_final_response_length", gen.sampling_params.max_tokens),
    )
    if any(type(value) is not int or value < 1 for value in (
        gen.num_trajectories, gen.val_config.num_trajectories, gen.max_prompt_length, max_output,
    )):
        raise ValueError("Task sampling counts and token limits must be positive integers")
    shared = {
        "generator.task": str(task_path),
        "trainer.max_prompt_length": gen.max_prompt_length,
        "generator.max_input_length": gen.max_prompt_length,
        "generator.n_samples_per_prompt": gen.num_trajectories,
        "generator.eval_n_samples_per_prompt": gen.val_config.num_trajectories,
        "generator.sampling_params.max_generate_length": max_output,
        "generator.eval_sampling_params.max_generate_length": max_output,
        "generator.sampling_params.temperature": gen.sampling_params.temperature,
        "generator.sampling_params.top_p": gen.sampling_params.top_p,
        "generator.eval_sampling_params.temperature": gen.val_config.sampling_params.temperature,
        "generator.eval_sampling_params.top_p": gen.val_config.sampling_params.top_p,
    }
    for key, expected in shared.items():
        if OmegaConf.select(overrides, key, default=expected) != expected:
            raise ValueError(f"{key} must match {task_path} ({expected!r}); change the task YAML instead")
        OmegaConf.update(overrides, key, expected, force_add=True)
    return OmegaConf.to_container(overrides, resolve=True)
