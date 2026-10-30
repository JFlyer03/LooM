import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from loom.integrations.verl.memory_config import prepare_memory_overrides


ROOT = Path(__file__).resolve().parents[1]


def parsed(args):
    return OmegaConf.from_dotlist([arg.lstrip("+") for arg in args if "=" in arg])


def test_verl_launcher_forwards_paths_and_maps_task_settings(tmp_path):
    capture = tmp_path / "args.json"
    executable = tmp_path / "python"
    executable.write_text(
        f"#!{sys.executable}\nimport os,sys,json\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({'args':sys.argv[1:],'cwd':os.getcwd()}))\n"
    )
    executable.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "CAPTURE": str(capture)}
    subprocess.run([
        "bash", str(ROOT / "examples/MemAgent/run_memagent_verl.sh"),
        "actor_rollout_ref.model.path=/local/model with spaces", "trainer.total_training_steps=1",
    ], env=env, cwd=tmp_path, check=True, capture_output=True)
    command = json.loads(capture.read_text())
    assert command["cwd"] == str(ROOT)
    assert command["args"][:2] == ["-m", "loom.integrations.verl.memory_main"]
    args = command["args"][2:]
    cfg = parsed(args)
    args += [f"+skyrl_agent.task_yaml={ROOT / cfg.skyrl_agent.task_yaml}"]
    cfg = parsed(prepare_memory_overrides(args))
    assert cfg.actor_rollout_ref.model.path == "/local/model with spaces"
    assert cfg.trainer.total_training_steps == 1
    assert cfg.algorithm.adv_estimator == "grpo"
    assert cfg.algorithm.norm_adv_by_std_in_grpo is False
    assert cfg.algorithm.use_kl_in_reward is False
    assert cfg.actor_rollout_ref.actor.loss_agg_mode == "token-mean"
    assert cfg.actor_rollout_ref.actor.kl_loss_coef == 0.001
    assert cfg.actor_rollout_ref.model.override_config._attn_implementation == "sdpa"
    assert cfg.actor_rollout_ref.rollout.n == cfg.skyrl_agent.num_trajectories == 16
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 4
    assert cfg.actor_rollout_ref.rollout.max_model_len == 8192
    assert cfg.actor_rollout_ref.rollout.mode == "sync"
    assert cfg.actor_rollout_ref.rollout.dtype == "float16"
    assert cfg.actor_rollout_ref.rollout.memory_task_yaml.endswith("memagent_verl.yaml")
    assert cfg.actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype == "fp32"
    assert cfg.data.train_files != cfg.data.val_files


def test_verl_task_is_source_for_sampling_and_budgets(tmp_path):
    task = OmegaConf.load(ROOT / "examples/MemAgent/memagent_verl.yaml")
    task.generator.num_trajectories = 3
    task.generator.val_config.num_trajectories = 2
    task.memory.max_memorization_length = 64
    task.memory.max_final_response_length = 128
    task.generator.max_prompt_length = 4096
    path = tmp_path / "task.yaml"
    OmegaConf.save(task, path)
    args = prepare_memory_overrides([f"+skyrl_agent.task_yaml={path}", "--cfg", "job", "--resolve"])
    assert args[-3:] == ["--cfg", "job", "--resolve"]
    cfg = parsed(args)
    assert cfg.actor_rollout_ref.rollout.n == 3
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 2
    assert cfg.data.max_response_length == 128
    assert cfg.actor_rollout_ref.rollout.max_model_len == 4224
    with pytest.raises(ValueError, match="must match"):
        prepare_memory_overrides([f"+skyrl_agent.task_yaml={path}", "actor_rollout_ref.rollout.n=8"])


def test_verl_entrypoint_rejects_skyrl_task():
    with pytest.raises(ValueError, match="infer_backend: verl"):
        prepare_memory_overrides([f"+skyrl_agent.task_yaml={ROOT / 'examples/MemAgent/memagent.yaml'}"])
