"""Convert MemAgent QA data to the standard VeRL dataset columns."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .long_context_memory import LongContextMemoryTask


def convert_records(records):
    for record in records:
        instance = LongContextMemoryTask.normalize_instance(record)
        yield {
            "prompt": LongContextMemoryTask.get_instruction(instance),
            "instance": instance,
            "data_source": "loom_memagent",
            "reward_model": {"style": "rule", "ground_truth": instance["answers"]},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.input.suffix == ".json":
        records = json.loads(args.input.read_text(encoding="utf-8"))
    elif args.input.suffix == ".jsonl":
        records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = pd.read_parquet(args.input).to_dict("records")
    if not isinstance(records, list) or not records:
        parser.error("Expected a nonempty JSON array, JSONL file, or Parquet table")

    dataset = pd.DataFrame(convert_records(records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
