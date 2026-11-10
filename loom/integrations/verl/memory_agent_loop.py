"""Reset-context memory steps on VERL's native V1/TransferQueue trainer.

VERL owns sampling groups, multi-output GRPO, parameter updates and weight sync.
Each independent memory context is an AgentLoopOutput in the same native session.
"""

from time import perf_counter

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput

from ..memory_episode import run_memory_episode


class MemoryAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        started = perf_counter()
        task_yaml = self.config.skyrl_agent.task_yaml
        if "instance" not in kwargs:
            raise ValueError("Memory rollout needs the dataset's full 'instance' column")
        result = await run_memory_episode(
            task_yaml, self.server_manager, self.tokenizer,
            {"instance": kwargs["instance"], "__loom_uid": str(kwargs["uid"]),
             "data_source": kwargs.get("data_source", "loom_memagent")},
            sampling_params,
        )
        # Native V1 names outputs uid_session_step, computes GRPO on the final
        # output once, and broadcasts that trajectory advantage to all its steps.
        outputs = []
        for i, response in enumerate(result["response_ids"]):
            outputs.append(AgentLoopOutput(
                prompt_ids=result["prompt_token_ids"][i],
                response_ids=response,
                response_mask=result["loss_masks"][i],
                response_logprobs=result["rollout_logprobs"][i],
                reward_score=float(result["traj_rewards"][0]),
                num_turns=len(result["response_ids"]),
                metrics=AgentLoopMetrics(generate_sequences=(perf_counter() - started) / len(result["response_ids"])),
                extra_fields={"reward_extra_info": {"acc": float(result["traj_rewards"][0])}},
            ))
        return outputs
