"""Keep full task records while rendering the dataset's string instructions."""

from verl.utils.dataset.rl_dataset import RLHFDataset


class MemoryDataset(RLHFDataset):
    def _build_messages(self, example, key):
        if isinstance(example[key], str):
            example = {**example, key: [{"role": "user", "content": example[key]}]}
        return super()._build_messages(example, key)
