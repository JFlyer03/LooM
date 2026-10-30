"""Launch MemAgent through the existing VeRL training entrypoint."""

import sys

from .memory_config import prepare_memory_overrides


def main():
    sys.argv[1:] = prepare_memory_overrides(sys.argv[1:])
    from .verl_main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
