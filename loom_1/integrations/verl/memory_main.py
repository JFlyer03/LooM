"""Launch MemAgent on VERL's native V1 trainer and multi-output AgentLoop."""

import os
import sys

from .memory_config import prepare_memory_overrides


def main():
    # shortcut: V2 warmup faults in this colocated profile; retry after upgrading vLLM.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    sys.argv[1:] = prepare_memory_overrides(sys.argv[1:])
    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
