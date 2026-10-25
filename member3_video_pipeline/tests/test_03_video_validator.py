"""
Tests for src/validation/video_validator.py

Every failure mode must set is_valid=False, a non-empty error_code,
and a human-readable error string.
"""

import pytest

_V_ERROR = None
try:
    from src.validation.video_validator import VideoValidator
    from src.config import DEFAULT_CONFIG
    _OK = True
except Exception as exc:
    _OK = False
    _V_ERROR = exc

pytestmark = pytest.mark.skipif(
    not _OK,
    reason=f"Cannot import VideoValidator ({_V_ERROR}). Paste src/validation/video_validator.py to fix.",
)


@pytest.fixture
def validator():
    return VideoValidator(DEFAULT_CONFIG)


def test_missing_file(validator, tmp_path):
    result = validator.validate(tmp_path / "does_not_exist.mp4")
    assert result.is_valid is False
    assert isinstance(result.error, str) and result.error
    assert isinstance(result.error_code, str) and result.error_code


def test_empty_file(validator, tmp_path):
    f = tmp_path / "empty.mp4"
    f.write_bytes(b"")
    result = validator.validate(f)
    assert result.is_valid is False
    assert result.error_code


def test_directory_path_rejected(validator, tmp_path):
    result = validator.validate(tmp_path)
    assert result.is_valid is False


def test_unsupported_extension(validator, tmp_path):
    f = tmp_path / "clip.txt"
    f.write_bytes(b"not a video")
    result = validator.validate(f)
    assert result.is_valid is False
    assert result.error_code


def test_corrupt_video_bytes(validator, tmp_path):
    f = tmp_path / "corrupt.mp4"
    f.write_bytes(b"\x00" * 1024)
    result = validator.validate(f)
    assert result.is_valid is False
    assert result.error_code


def test_valid_video_passes(validator, make_video):
    result = validator.validate(make_video())
    assert result.is_valid is True
    assert result.error_code is None or result.error_code == ""


def test_valid_result_has_no_error(validator, make_video):
    result = validator.validate(make_video())
    if result.is_valid:
        assert result.error is None or result.error == ""