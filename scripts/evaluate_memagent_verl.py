"""Re-score native VERL validation dumps, counting only final MemAgent answers."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from loom.tasks.verifiers.memagent import compute_score


def score_dump(path, references, samples_per_question):
    scores = defaultdict(list)
    row_count = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        row_count += 1
        prompt = row["input"]
        if "You are presented with a problem and a previous memory." not in prompt:
            continue
        match = re.search(r"<problem>\s*(.*?)\s*</problem>", prompt, re.DOTALL)
        if match is None or match[1] not in references:
            raise ValueError("Final answer does not match the held-out validation questions")
        question = match[1]
        answers = references[question]
        if row.get("gts") != answers:
            raise ValueError(f"Missing or inconsistent validation references: {question}")
        score = compute_score(row["output"], answers)
        if score != row["score"]:
            raise ValueError(f"Logged reward differs from re-scored final answer: {question}")
        scores[question].append(score)
    if set(scores) != set(references):
        raise ValueError(f"Incomplete validation: {len(scores)} of {len(references)} questions")
    if any(len(values) != samples_per_question for values in scores.values()):
        raise ValueError("Validation sampling count differs from the task YAML")
    means = np.array([np.mean(scores[q]) for q in sorted(references)])
    return {
        "source": str(path.resolve()),
        "dump_rows": row_count,
        "questions": len(scores),
        "final_answers": len(scores) * samples_per_question,
        "samples_per_question": samples_per_question,
        "accuracy": float(means.mean()),
        "any_correct_in_group": float(np.mean([max(v) for v in scores.values()])),
        "all_logged_scores_verified": True,
    }, means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--samples-per-question", type=int, default=4)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    instances = pd.read_parquet(args.validation_data)["instance"]
    references = {r["question"].strip(): list(r["answers"]) for r in instances}
    if not references or len(references) != len(instances):
        raise ValueError("Validation questions must be nonempty and unique")
    report, means = score_dump(args.generations, references, args.samples_per_question)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(means), size=(10000, len(means)))
    report["accuracy_95pct_question_bootstrap_ci"] = np.quantile(
        means[draws].mean(axis=1), [.025, .975],
    ).tolist()
    if args.baseline:
        baseline, baseline_means = score_dump(args.baseline, references, args.samples_per_question)
        delta = means - baseline_means
        report["baseline"] = baseline
        report["accuracy_delta"] = float(delta.mean())
        report["delta_95pct_paired_question_bootstrap_ci"] = np.quantile(
            delta[draws].mean(axis=1), [.025, .975],
        ).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
