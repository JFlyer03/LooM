"""Configure memory rollout, then call the pinned upstream slime training entrypoint."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[3]


def prepare_args(args, task_path):
    task = OmegaConf.load(task_path)
    if task.agent_cls != "loom.agents.memory.MemAgent" or task.generator.infer_backend != "slime":
        raise ValueError("The slime entrypoint requires a MemAgent task with infer_backend: slime")
    output_length = max(task.memory.max_memorization_length, task.memory.max_final_response_length)
    if any(type(value) is not int or value < 1 for value in (
        task.generator.num_trajectories, task.generator.val_config.num_trajectories,
        task.generator.max_prompt_length, output_length,
    )) or task.generator.num_trajectories < 2:
        raise ValueError("Token limits must be positive integers and GRPO needs at least two trajectories")
    shared = {
        "--custom-generate-function-path": "loom.integrations.slime.memory_rollout.generate",
        "--n-samples-per-prompt": task.generator.num_trajectories,
        "--n-samples-per-eval-prompt": task.generator.val_config.num_trajectories,
        "--rollout-temperature": task.generator.sampling_params.temperature,
        "--rollout-top-p": task.generator.sampling_params.top_p,
        "--eval-temperature": task.generator.val_config.sampling_params.temperature,
        "--eval-top-p": task.generator.val_config.sampling_params.top_p,
        "--sglang-context-length": task.generator.max_prompt_length + output_length,
        "--rollout-max-response-len": output_length,
        "--advantage-estimator": "grpo",
        "--metadata-key": "metadata",
        "--sglang-grammar-backend": "xgrammar",
    }
    if "--disable-rewards-normalization" in args:
        raise ValueError("Memory GRPO requires trajectory-level reward centering")
    additions = []
    if "--disable-grpo-std-normalization" not in args:
        additions.append("--disable-grpo-std-normalization")
    for flag, value in shared.items():
        supplied = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == flag]
        supplied += [arg.split("=", 1)[1] for arg in args if arg.startswith(flag + "=")]
        if any(str(candidate) != str(value) for candidate in supplied):
            raise ValueError(f"{flag} must match {task_path}: {value}")
        if not supplied:
            additions.extend([flag, str(value)])
    return [*additions, *args]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--loom-task", type=Path, default=ROOT / "examples/MemAgent/memagent_slime.yaml")
    parser.add_argument("--slime-source", type=Path, default=ROOT / ".backend-src/slime")
    parser.add_argument("--print-args", action="store_true")
    options, rest = parser.parse_known_args()
    options.loom_task = options.loom_task.resolve()
    args = prepare_args(rest, options.loom_task)
    if options.print_args:
        print(json.dumps(args, indent=2))
        return
    entrypoint = options.slime_source / "train.py"
    if not entrypoint.is_file():
        raise FileNotFoundError("Prepare the pinned sources with scripts/prepare_unified_sources.py")
    spec = importlib.util.spec_from_file_location("_loom_slime_train", entrypoint)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv[1:] = args
    native_args, restore_plan = module.parse_args(return_restore_plan=True)
    native_args.loom_task = str(options.loom_task)
    module.configure_logger()
    module.main(native_args, restore_plan)


if __name__ == "__main__":
    main()
