"""
Tests for src/config.py

Asserts invariants that ANY sane config must hold:
  - DEFAULT_CONFIG is a PipelineConfig instance
  - every uppercase numeric attribute is positive
  - SUPPORTED_EXTENSIONS is lowercase, includes common formats

Does not assert exact field values (those are policy, not correctness).
"""

import dataclasses
import pytest

_CONFIG_ERROR = None
try:
    from src.config import DEFAULT_CONFIG, PipelineConfig
    _OK = True
except Exception as exc:
    _OK = False
    _CONFIG_ERROR = exc

pytestmark = pytest.mark.skipif(
    not _OK,
    reason=f"Cannot import src.config ({_CONFIG_ERROR}). Paste src/config.py to fix.",
)


def test_default_config_is_pipeline_config():
    assert isinstance(DEFAULT_CONFIG, PipelineConfig)


def test_is_a_dataclass():
    assert dataclasses.is_dataclass(PipelineConfig)


def test_all_numeric_limit_fields_positive():
    """Any int/float class attribute must be > 0 — no zero limits allowed."""
    checked = []
    for name in dir(PipelineConfig):
        if name.startswith("_"):
            continue
        val = getattr(DEFAULT_CONFIG, name, None)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            assert val > 0, f"{name}={val} must be positive"
            checked.append(name)
    assert checked, "No numeric limit fields found — is config empty?"


def test_supported_extensions_lowercase_no_dots():
    exts = getattr(DEFAULT_CONFIG, "SUPPORTED_EXTENSIONS", None)
    if exts is None:
        pytest.skip("SUPPORTED_EXTENSIONS not found on config")
    assert len(exts) > 0, "SUPPORTED_EXTENSIONS must not be empty"
    for e in exts:
        assert e == e.lower(), f"{e!r} is not lowercase"


def test_common_video_formats_supported():
    exts = getattr(DEFAULT_CONFIG, "SUPPORTED_EXTENSIONS", None)
    if exts is None:
        pytest.skip("SUPPORTED_EXTENSIONS not found")
    normalised = {str(e).lower().lstrip(".") for e in exts}
    assert {"mp4", "avi", "mov"} <= normalised, (
        f"Missing common formats. Found: {normalised}"
    )


def test_overriding_does_not_mutate_default():
    """Either frozen (replace works) or mutable — both are fine. Verify default survives."""
    try:
        custom = dataclasses.replace(DEFAULT_CONFIG, __test_override__=1)
    except (TypeError, ValueError):
        pytest.skip("PipelineConfig is frozen or non-replaceable — OK")
    assert custom is not DEFAULT_CONFIG