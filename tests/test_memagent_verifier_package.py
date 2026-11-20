"""Package reorganization must preserve MemAgent's original reward semantics."""

import pytest

from loom.tasks.verifiers.memagent import compute_score
from loom.tasks.verifiers.memagent.grader import compute_score as implementation


@pytest.mark.parametrize(
    "response, references, expected",
    [
        (r"Reasoning. \boxed{Paris}", ["Paris"], 1.0),
        ("Paris", ["Paris"], 0.0),
        (r"\boxed{wrong} then \boxed{Paris}", ["Paris"], 1.0),
        (r"\boxed{Paris}", ["London", "Paris"], 1.0),
        (r"\boxed{The Eiffel Tower}", ["Eiffel Tower"], 0.0),
        (r"\boxed{Paris}" + "x" * 300, ["Paris"], 0.0),
    ],
)
def test_package_exports_original_memagent_grader(response, references, expected):
    assert compute_score is implementation
    assert compute_score(response, references) == expected
