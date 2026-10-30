import asyncio
from copy import deepcopy

from ..base import AgentRunner, BaseTrajectory
from ...functional.utils import transitions_to_training_data


class MemoryTrajectory(BaseTrajectory):
    async def initialize_trajectory(self):
        self.instance = await self.task.initialize_runtime(self.data["instance"])
        self.agent = self.agent_cls(self.cfg, self.infer_engine, self.tokenizer)

    async def generate_trajectory(self):
        if hasattr(self.agent, "prepare_shared"):
            shared = self.shared_preparation
            if "task" not in shared:
                shared["task"] = asyncio.create_task(self.agent.prepare_shared(self.instance))
            prepared = deepcopy(await shared["task"])
            finish_reason, result = await self.agent.run(self.instance, prepared=prepared)
        else:
            finish_reason, result = await self.agent.run(self.instance)
        self.result = {
            "instance_id": self.cfg.instance_id,
            "trajectory_id": self.cfg.trajectory_id,
            "messages": self.agent.messages,
            "transitions": self.agent.transitions,
            "results": self.task.complete_runtime(result),
            "state": {"memory_history": list(self.agent.memory_history)},
            "finish_reason": finish_reason,
            "finish": finish_reason == "completed",
        }

    async def evaluate_trajectory(self):
        reward = 0.0
        if self.result["finish"]:
            reward = await self.task.evaluate_result(self.result["results"], self.instance, self.cfg.memory)
        if isinstance(reward, dict):
            self.result["stage_rewards"] = reward
            selected = (self.cfg.memory or {}).get("trainable_stages", list(reward))
            if not selected or any(stage not in reward for stage in selected):
                raise ValueError("trainable_stages must name evaluated memory stages")
            for transition in self.agent.transitions:
                transition.reward = float(reward[transition.metrics["stage"]])
                transition.metrics["stage_reward"] = transition.reward
            self.result["separate_stage_rewards"] = len(selected) > 1
            reward = sum(reward[stage] for stage in selected) / len(selected)
        self.result["reward"] = float(reward)
        if "stage_rewards" not in self.result:
            self.agent.transitions[-1].reward = float(reward)


class MemoryAgentRunner(AgentRunner):
    """Reuse scheduling/backends; avoid the upstream SWE result fallback.

    Each reset context is a separate training row with the same trajectory key.
    """

    def _initialize_trajectories(self, val_mode=False):
        super()._initialize_trajectories(val_mode=val_mode)
        for trajectories in self.trajectories.values():
            shared = {}
            for trajectory in trajectories.values():
                trajectory.shared_preparation = shared

    async def run(self, input_batch, val_mode=False):
        if getattr(self, "_running", False):
            raise RuntimeError("One MemoryAgentRunner cannot process two batches concurrently")
        self._running = True
        try:
            return await super().run(input_batch, val_mode=val_mode)
        finally:
            self.trajectories = {}
            self.batch = []
            self._running = False

    def _post_process_results(self, return_tensors=False, val_mode=False):
        output = {key: [] for key in (
            "prompt_token_ids", "response_ids", "rewards", "loss_masks", "stop_reasons",
            "rollout_logprobs", "traj_rewards", "episode_nums", "is_last_episode", "traj_idx", "trajectory_keys", "env_metrics",
        )}
        all_have_logprobs = True
        for instance_id, trajectories in self.trajectories.items():
            for repetition_id, trajectory in trajectories.items():
                result = trajectory.result
                transitions = result["transitions"]
                if not transitions:
                    raise ValueError("Memory trajectory produced no model calls")
                # Convert separately: never merge two independent memory contexts,
                # even if their token sequences happen to share a prefix.
                rows = [transitions_to_training_data([transition])[0] for transition in transitions]
                reward = float(result["reward"])
                valid = result["finish_reason"] == "completed"
                output["traj_rewards"].append(reward)
                output["episode_nums"].append(len(rows))
                for i, (row, transition) in enumerate(zip(rows, transitions)):
                    output["prompt_token_ids"].append(row.input_tokens)
                    output["response_ids"].append(row.response_tokens)
                    trainable = transition.metrics.get("trainable", True)
                    output["loss_masks"].append([int(valid and trainable)] * len(row.response_tokens))
                    output["rewards"].append(
                        float(transition.reward) if result.get("separate_stage_rewards") else reward
                    )
                    output["stop_reasons"].append(transition.metrics["stop_reason"])
                    output["rollout_logprobs"].append(row.response_logprobs)
                    all_have_logprobs &= transition.ac.logprobs is not None
                    output["is_last_episode"].append(i == len(rows) - 1)
                    output["traj_idx"].append(f"{instance_id}-traj{repetition_id}")
                    output["trajectory_keys"].append((str(instance_id), repetition_id))
                    output["env_metrics"].append({
                        **transition.metrics,
                        "finish_reason": result["finish_reason"],
                    })
        if not all_have_logprobs:
            output["rollout_logprobs"] = None
        count = len(output["traj_rewards"])
        output["rollout_metrics"] = {
            "memory/num_trajectories": count,
            "memory/num_model_calls": len(output["response_ids"]),
            "memory/mean_reward": sum(output["traj_rewards"]) / count if count else 0.0,
        }
        return output
