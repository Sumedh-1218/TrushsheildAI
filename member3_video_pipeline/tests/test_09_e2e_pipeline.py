"""
Phase 2: End-to-End Integration Test for src/pipeline.py
"""

import json
import pytest
import dataclasses

from src.pipeline import analyze_video
from src.config import DEFAULT_CONFIG


def test_pipeline_happy_path(make_video):
    """
    Run the entire pipeline on a real (dummy) video file.
    """
    # 1. Generate a real 10-frame video file (10 fps = 1 second total)
    video_path = make_video(frames=10, fps=10.0)
    
    # 2. Create a custom config that samples EVERY frame (0.1s interval)
    # Otherwise, the default 1.0s interval will only grab 1 frame from a 1s video.
    test_config = dataclasses.replace(DEFAULT_CONFIG, SAMPLING_INTERVAL_SECONDS=0.1)
    
    # 3. Run the full pipeline with our custom config
    result = analyze_video(str(video_path), config=test_config)
    
    # 4. Assert high-level status
    assert isinstance(result, dict)
    assert result["status"] == "completed"
    
    # 5. Assert video metadata was extracted correctly
    video_meta = result.get("video", {})
    assert video_meta["width"] == 64
    assert video_meta["height"] == 64
    assert video_meta["fps"] == 10.0
    
    # 6. Assert frame processing occurred
    # Now it should sample all 10 frames. They have no faces, so they are unscorable.
    assert result["frames_unscorable"] == 10
    assert result["frames_analyzed"] == 0
    assert result["score"] is None 
    
    # 7. Assert the label logic handled the empty score correctly
    assert result["label"] == "inconclusive"
    
    # 8. Verify it is fully JSON serializable
    try:
        json_str = json.dumps(result)
        assert len(json_str) > 0
    except TypeError:
        pytest.fail("Pipeline result is not JSON serializable!")


def test_pipeline_fails_gracefully_on_bad_file(tmp_path):
    """
    Ensure the pipeline catches errors and returns a 'failed' dict.
    """
    bad_path = tmp_path / "this_does_not_exist.mp4"
    result = analyze_video(str(bad_path))
    
    assert result["status"] == "failed"
    assert result["label"] == "error"
    assert "error" in result


def test_pipeline_flags_mock_detector(make_video):
    """
    Ensure the pipeline adds the 'MOCK DETECTOR' warning to limitations.
    """
    video_path = make_video(frames=2)
    result = analyze_video(str(video_path))
    
    limitations = result.get("limitations", [])
    assert any("MOCK DETECTOR" in lim for lim in limitations)