from pathlib import Path
import json
import os
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from loom)1.integrations.slime.memory_main import prepare_args
from loom)1.integrations.slime.prepare_data import convert_for_slime


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "examples/MemAgent/memagent_slime.yaml"


def test_slime_keeps_full_task_in_metadata_for_each_input():
    records = [{"context": "ABCDEF", "question": "Country?", "answers": ["France"]}]
    rows = convert_for_slime(records)
    assert len(rows) == 1
    assert rows[0]["metadata"]["instance"]["context"] == "ABCDEF"
    assert rows[0]["metadata"]["instance"]["answers"] == ["France"]
    assert records[0]["context"] == "ABCDEF"


def test_slime_sampling_group_is_owned_by_native_backend(tmp_path):
    cfg = OmegaConf.load(TASK)
    cfg.generator.num_trajectories = 3
    cfg.memory.max_memorization_length = 48
    cfg.memory.max_final_response_length = 96
    path = tmp_path / "task.yaml"
    OmegaConf.save(cfg, path)
    args = prepare_args(["--hf-checkpoint", "/model with spaces"], path)
    assert args[args.index("--n-samples-per-prompt") + 1] == "3"
    assert args[args.index("--rollout-max-response-len") + 1] == "96"
    assert args[-2:] == ["--hf-checkpoint", "/model with spaces"]
    assert args[args.index("--custom-generate-function-path") + 1].endswith("memory_rollout.generate")
    with pytest.raises(ValueError, match="must match"):
        prepare_args(["--n-samples-per-prompt=4"], path)
    with pytest.raises(ValueError, match="must match"):
        prepare_args(["--advantage-estimator", "gae"], path)


def test_finetuning_script_starts_at_zero_and_uses_selected_model_config(tmp_path):
    model_config = tmp_path / "model.sh"
    model_config.write_text("MODEL_ARGS=(--num-layers 1)\n")
    command = subprocess.run(["bash", str(ROOT / "examples/MemAgent/run_memagent_slime.sh"), "--print-args"],
                             env={**os.environ, "CONDA_PREFIX": sys.prefix, "PATH": "/usr/bin:/bin",
                                  "MODEL_CONFIG": str(model_config), "TASK_YAML": str(TASK),
                                  "HF_CHECKPOINT": "/model with spaces", "MEGATRON_CHECKPOINT": "/converted",
                                  "TRAIN_DATA": "/data.jsonl", "NUM_ROLLOUT": "2", "NUM_GPUS": "1"},
                             capture_output=True, text=True, check=True)
    args = json.loads(command.stdout)
    assert args[args.index("--start-rollout-id") + 1] == "0"
    assert args[args.index("--num-rollout") + 1] == "2"
    assert args[args.index("--num-layers") + 1] == "1"
    assert args[args.index("--hf-checkpoint") + 1] == "/model with spaces"


@pytest.mark.parametrize("field,value", [("num_trajectories", 1), ("num_trajectories", 2.5), ("max_prompt_length", 0)])
def test_slime_rejects_invalid_task_limits_before_starting_native_workers(tmp_path, field, value):
    cfg = OmegaConf.load(TASK)
    cfg.generator[field] = value
    path = tmp_path / "task.yaml"
    OmegaConf.save(cfg, path)
    with pytest.raises(ValueError, match="positive integers"):
        prepare_args([], path)
