"""Convert the existing task Parquet to slime JSONL without dropping task fields."""

import argparse
import json
from pathlib import Path

from loom.tasks.prepare_memory_data import convert_records


def convert_for_slime(records):
    return [{"prompt": row["prompt"], "metadata": {"instance": row["instance"],
             "original_index": row["original_index"]}} for row in convert_records(records)]


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path to preserve data")
    rows = pd.read_parquet(args.input).to_dict("records")
    rows = [row.get("instance", row) for row in rows]
    result = convert_for_slime(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in result:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(result)} instances to {args.output}")


if __name__ == "__main__":
    main()
