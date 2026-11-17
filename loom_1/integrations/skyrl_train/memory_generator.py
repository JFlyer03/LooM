"""Translate native SkyRL sample IDs to the memory runner and back.

No loss, optimizer, reward normalization or weight synchronization lives here.
"""

from copy import deepcopy

from ... import AutoAgentRunner


class MemoryGenerator:
    def __init__(self, generator_cfg, inference_engine_client, tokenizer):
        if not generator_cfg.step_wise_trajectories:
            raise ValueError("MemoryGenerator requires generator.step_wise_trajectories=true")
        self.runner = AutoAgentRunner.from_task(
            generator_cfg.task, infer_engine=inference_engine_client, tokenizer=tokenizer,
        )
        if self.runner.cfg.agent_cls.endswith("MemoryR1Agent"):
            stages = self.runner.cfg.memory.get("trainable_stages", ["extraction", "update"])
            if len(stages) != 1:
                raise ValueError(
                    "Native step-wise SkyRL broadcasts one advantage per trajectory. "
                    "For Memory-R1 select one trainable_stages entry per run; joint training "
                    "with separate extraction/update advantages needs a multi-objective adapter."
                )

    async def generate(self, input_batch):
        ids = input_batch.get("trajectory_ids")
        extras = input_batch.get("env_extras")
        if not ids or extras is None or len(ids) != len(extras):
            raise ValueError("Native SkyRL trajectory_ids and env_extras must align")
        trajectory_id_cls = type(ids[0])
        instances = {}
        repetitions = {}
        for tid, extra in zip(ids, extras):
            if tid.instance_id not in instances:
                instances[tid.instance_id] = {**deepcopy(extra), "__loom_uid": tid.instance_id}
                repetitions[tid.instance_id] = []
            repetitions[tid.instance_id].append(tid.repetition_id)
        val_mode = input_batch["batch_metadata"].training_phase == "eval"
        cfg = self.runner.cfg.generator.val_config if val_mode else self.runner.cfg.generator
        expected = list(range(cfg.num_trajectories))
        if any(reps != expected for reps in repetitions.values()):
            raise ValueError("SkyRL samples-per-prompt must match the task YAML num_trajectories in order")
        # Honor the trainer's sampling policy without altering the task's memory settings.
        if input_batch.get("sampling_params") is not None:
            cfg.sampling_params = deepcopy(input_batch["sampling_params"])
        result = await self.runner.run({"env_extras": list(instances.values())}, val_mode=val_mode)
        return {
            **{key: result[key] for key in (
                "prompt_token_ids", "response_ids", "loss_masks", "rewards", "stop_reasons",
                "rollout_logprobs", "rollout_metrics",
            )},
            "is_last_step": result["is_last_episode"],
            "trajectory_ids": [trajectory_id_cls(uid, repetition) for uid, repetition in result["trajectory_keys"]],
            "env_metrics": result["env_metrics"],
        }
