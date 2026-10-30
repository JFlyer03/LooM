"""MemAgent on SkyRL's native trainer and native step-wise advantage handling."""

import sys
from dataclasses import dataclass

import ray
from skyrl.train.config import GeneratorConfig, make_config
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import initialize_ray, validate_cfg

from .memory_generator import MemoryGenerator
from .memory_config import DEFAULT_TASK, prepare_memory_config


@dataclass
class MemoryGeneratorConfig(GeneratorConfig):
    task: str = DEFAULT_TASK


MemoryTrainConfig = make_config(generator_cls=MemoryGeneratorConfig)


class MemoryExperiment(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return MemoryGenerator(cfg.generator, inference_engine_client, tokenizer)


@ray.remote(num_cpus=1)
def run_experiment(cfg):
    MemoryExperiment(cfg).run()


def main():
    cfg = MemoryTrainConfig.from_cli_overrides(prepare_memory_config(sys.argv[1:]))
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(run_experiment.remote(cfg))


if __name__ == "__main__":
    main()
