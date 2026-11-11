import json
import runpy
from pathlib import Path

import pytest


score_dump = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/evaluate_memagent_verl.py"),
)["score_dump"]


def test_validation_recounts_final_answers_without_counting_memory_steps(tmp_path):
    memory = {"input": "Read this <section>memory</section>", "output": "Paris", "score": 1}
    answer = {
        "input": "You are presented with a problem and a previous memory.\n<problem>Capital?</problem>",
        "output": r"\boxed{Paris}", "gts": ["Paris"], "score": 1,
    }
    path = tmp_path / "validation.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [memory, answer, memory, answer]))
    report, means = score_dump(path, {"Capital?": ["Paris"]}, 2)
    assert report["dump_rows"] == 4
    assert report["final_answers"] == 2
    assert report["questions"] == 1
    assert means.tolist() == [1.0]

    with pytest.raises(ValueError, match="sampling count"):
        score_dump(path, {"Capital?": ["Paris"]}, 4)
    with pytest.raises(ValueError, match="Incomplete validation"):
        score_dump(path, {"Capital?": ["Paris"], "Other?": ["London"]}, 2)


def test_validation_rejects_incorrect_logged_rewards(tmp_path):
    row = {
        "input": "You are presented with a problem and a previous memory.\n<problem>Capital?</problem>",
        "output": r"\boxed{London}", "gts": ["Paris"], "score": 1,
    }
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Logged reward differs"):
        score_dump(path, {"Capital?": ["Paris"]}, 1)
