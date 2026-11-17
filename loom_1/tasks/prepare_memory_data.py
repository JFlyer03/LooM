"""Convert legacy loom QA JSON/Parquet to SkyRL's prompt + env_extras columns."""

import argparse
import json
from pathlib import Path

from .memory import LongContextMemoryTask


def convert_records(records, task="longcontext"):
    task_cls = LongContextMemoryTask
    if task != "longcontext":
        from .memory_bank import MemoryBankTask, RerankBankTask
        task_cls = {"memory_bank": MemoryBankTask, "rerank_bank": RerankBankTask}[task]
    result = []
    for index, record in enumerate(records):
        instance = task_cls.normalize_instance(record)
        result.append({
            "prompt": task_cls.get_instruction(instance),
            "instance": instance,
            "data_source": "loom_memagent" if task == "longcontext" else f"loom_{task}",
            "original_index": str(record.get("index", index)),
        })
    return result


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["longcontext", "memory_bank", "rerank_bank"], default="longcontext")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from the source dataset")
    if args.output.exists():
        parser.error("Output exists; choose a new path to preserve existing datasets")
    if args.input.suffix == ".jsonl":
        records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif args.input.suffix == ".json":
        records = json.loads(args.input.read_text(encoding="utf-8"))
    else:
        records = pd.read_parquet(args.input).to_dict("records")
    if not isinstance(records, list) or not records:
        parser.error("Expected a nonempty JSON array or Parquet table")
    data = convert_records(records, task=args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(args.output, index=False)
    print(f"Saved {len(data)} instances to {args.output}; no filtering or train/test split performed")


if __name__ == "__main__":
    main()
