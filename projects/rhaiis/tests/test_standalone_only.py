from types import SimpleNamespace

import pytest

from projects.core.library import config
from projects.rhaiis.orchestration.test_phase import _standalone_only_analysis


@pytest.mark.parametrize(
    "run_benchmark,profiler_enabled,expected",
    [
        (False, False, True),
        (True, False, False),
        (False, True, False),
    ],
)
def test_standalone_only_analysis(
    monkeypatch, run_benchmark: bool, profiler_enabled: bool, expected: bool
) -> None:
    values = {
        "tests.rhaiis.run_benchmark": run_benchmark,
        "rhaiis.profiler": {"enabled": profiler_enabled},
        "rhaiis.optional_phase.excluded_workloads": [],
    }
    monkeypatch.setattr(
        config,
        "project",
        SimpleNamespace(get_config=lambda path, default=None, **kwargs: values.get(path, default)),
    )

    assert _standalone_only_analysis(["profile1"]) is expected
