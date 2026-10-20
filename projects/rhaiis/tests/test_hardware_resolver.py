import sys
from pathlib import Path
from types import SimpleNamespace

from projects.rhaiis.orchestration.runtime_config import apply_hardware_spec

_ORCHESTRATION_DIR = Path(__file__).resolve().parents[1] / "orchestration"


def test_sets_gpu_count_from_tp_size():
    spec = {}
    result = apply_hardware_spec(spec, tp_size=4, pp_size=1, dp_size=1, gpu_type="h200")
    assert result["gpuCount"] == 4
    assert result["gpuType"] == "h200"


def test_tp_times_pp_gives_total_gpu_count():
    spec = {}
    result = apply_hardware_spec(spec, tp_size=4, pp_size=2, dp_size=1, gpu_type="h200")
    assert result["gpuCount"] == 8


def test_tp_times_dp_gives_total_gpu_count():
    spec = {}
    result = apply_hardware_spec(spec, tp_size=8, pp_size=1, dp_size=2, gpu_type="h200")
    assert result["gpuCount"] == 16


def test_tp_times_pp_times_dp_gives_total_gpu_count():
    spec = {}
    result = apply_hardware_spec(spec, tp_size=4, pp_size=2, dp_size=2, gpu_type="h200")
    assert result["gpuCount"] == 16


def test_overrides_wrong_gpu_count_when_gpu_type_already_set():
    # Regression: early-return on gpuType caused submitted gpuCount=1 to be silently kept
    spec = {"gpuCount": 1, "gpuType": "h200"}
    result = apply_hardware_spec(spec, tp_size=8, pp_size=1, dp_size=1, gpu_type="h200")
    assert result["gpuCount"] == 8


def test_all_defaults_gives_gpu_count_1():
    spec = {}
    result = apply_hardware_spec(spec, tp_size=1, pp_size=1, dp_size=1, gpu_type="h200")
    assert result["gpuCount"] == 1


def test_no_gpu_type_returns_empty():
    spec = {"gpuCount": 4}
    result = apply_hardware_spec(spec, tp_size=4, pp_size=1, dp_size=1, gpu_type=None)
    assert result == {}


def test_resolver_overrides_gpu_count_when_gpu_type_supplied(monkeypatch):
    # Regression: the resolver used to return early when gpuType was already set,
    # silently keeping the submitted gpuCount instead of deriving it from the TP size.
    # ci.py imports its sibling modules top-level, as when run as a script.
    if str(_ORCHESTRATION_DIR) not in sys.path:
        sys.path.append(str(_ORCHESTRATION_DIR))

    from projects.core.library import config
    from projects.rhaiis.orchestration import ci, runtime_config

    monkeypatch.setattr(ci, "test_rhaiis", SimpleNamespace(init=lambda: None))
    monkeypatch.setattr(
        config, "project", SimpleNamespace(get_config=lambda _key, _default=None: {})
    )
    monkeypatch.setattr(runtime_config, "get_test_model_key", lambda: "test-model")
    monkeypatch.setattr(
        runtime_config, "get_model", lambda _key: {"vllm_args": {"tensor-parallel-size": 8}}
    )
    monkeypatch.setattr(runtime_config, "get_engine", lambda: "vllm")
    monkeypatch.setattr(runtime_config, "get_accelerator", lambda: "h200")
    monkeypatch.setattr(runtime_config, "get_gpu_type", lambda _accelerator: None)

    result = ci.resolve_hardware_request({"gpuType": "h200", "gpuCount": 1})

    assert result["gpuCount"] == 8
    assert result["gpuType"] == "h200"
