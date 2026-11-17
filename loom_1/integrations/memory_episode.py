"""One independently sampled memory trajectory; backends own sampling groups."""

from copy import deepcopy

from loom)1 import AutoAgentRunner


async def run_memory_episode(task_yaml, infer_engine, tokenizer, record, sampling_params, *, val_mode=False):
    # A fresh runner also isolates concurrent backend sessions and their state.
    runner = AutoAgentRunner.from_task(str(task_yaml), infer_engine, tokenizer)
    if not runner.cfg.agent_cls.startswith("loom.agents.memory."):
        raise ValueError("The memory rollout adapter requires a memory agent")
    if len(runner.cfg.memory.get("trainable_stages", ["memory_update"])) != 1:
        raise ValueError("Independent stage advantages require a separate multi-objective adapter")
    for cfg in (runner.cfg.generator, runner.cfg.generator.val_config):
        cfg.num_trajectories = 1
        cfg.sampling_params = {**dict(cfg.sampling_params), **deepcopy(sampling_params)}
    result = await runner.run([deepcopy(record)], val_mode=val_mode)
    if not result["response_ids"] or any(not tokens for tokens in result["response_ids"]) or len(result["traj_rewards"]) != 1:
        raise ValueError("Expected exactly one nonempty memory trajectory")
    return result
