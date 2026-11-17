"""Keep full task records while rendering the dataset's string instructions."""

from verl.utils.dataset.rl_dataset import RLHFDataset


class MemoryDataset(RLHFDataset):
    def __getitem__(self, item):
        example = super().__getitem__(item)
        # VERL's validation dumps use this field to retain the reference answers.
        example.setdefault("reward_model", {
            "style": "rule", "ground_truth": list(example["instance"]["answers"]),
        })
        return example

    def _build_messages(self, example, key):
        if isinstance(example[key], str):
            example = {**example, key: [{"role": "user", "content": example[key]}]}
        return super()._build_messages(example, key)
