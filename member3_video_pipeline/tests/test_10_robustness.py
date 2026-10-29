"""
Phase 3: Robustness Tests
Tests the pipeline's ability to handle absolute garbage data without crashing.
"""

import pytest
from src.pipeline import analyze_video
from src.validation.video_validator import validate_video
from src.config import DEFAULT_CONFIG


def test_pipeline_handles_empty_file(tmp_path):
    """An empty file (0 bytes) should fail validation gracefully."""
    empty_file = tmp_path / "empty.mp4"
    empty_file.touch()  # Creates a 0-byte file
    
    result = analyze_video(str(empty_file))
    
    assert result["status"] == "failed"
    assert result["label"] == "error"
    assert "error" in result


def test_pipeline_handles_fake_mp4(tmp_path):
    """A file with .mp4 extension but random text inside should fail."""
    fake_video = tmp_path / "fake.mp4"
    fake_video.write_text("This is just a text file pretending to be a video.")
    
    result = analyze_video(str(fake_video))
    
    assert result["status"] == "failed"
    assert result["label"] == "error"
    assert "error" in result


def test_pipeline_handles_unsupported_extension(tmp_path):
    """A text file should be rejected immediately by the validator."""
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello world")
    
    # Check validator directly
    validation = validate_video(str(text_file), DEFAULT_CONFIG)
    assert validation.is_valid is False
    assert validation.error_code == "UNSUPPORTED_EXTENSION"


def test_pipeline_handles_missing_file():
    """A path that doesn't exist should not throw an unhandled exception."""
    result = analyze_video("C:/this/path/does/not/exist/video.mp4")
    assert result["status"] == "failed"
    assert "File not found" in result["error"] or "FILE_NOT_FOUND" in result["error"]