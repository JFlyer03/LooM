from ..base import BackendSpec, GeneratorInput, GeneratorOutput, register_backend
from .slime_backend import SlimeBackend


class SlimeGeneratorInput(GeneratorInput):
    def __init__(self, input_batch):
        self.input_batch = input_batch


class SlimeGeneratorOutput(GeneratorOutput):
    def __init__(self, result):
        self.result = result


register_backend("slime", BackendSpec(SlimeBackend, SlimeGeneratorOutput, SlimeGeneratorInput))
