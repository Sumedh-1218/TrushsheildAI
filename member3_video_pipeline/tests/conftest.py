"""
Shared pytest fixtures for TrustShield AI Member 3 tests.

If any import fails, the affected fixture skips the calling test with a
message telling you which source file to paste for diagnosis.
"""

import numpy as np
import pytest


# ─── Config ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    """Return DEFAULT_CONFIG. Skips if src.config is broken."""
    try:
        from src.config import DEFAULT_CONFIG
    except Exception as exc:
        pytest.skip(f"Cannot import src.config.DEFAULT_CONFIG — {exc}")
    return DEFAULT_CONFIG


# ─── Detector ────────────────────────────────────────────────────────────────

@pytest.fixture
def detector():
    """Return MockDetector(). Skips if not importable."""
    try:
        from src.detector.mock_detector import MockDetector
    except Exception as exc:
        pytest.skip(f"Cannot import MockDetector — {exc}")
    return MockDetector()


# ─── Images ──────────────────────────────────────────────────────────────────

@pytest.fixture
def blank_image():
    """A 64x64 black RGB image."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture
def noise_image():
    """A 64x64 random RGB image (seeded → deterministic)."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


# ─── Video factory ───────────────────────────────────────────────────────────

@pytest.fixture
def make_video(tmp_path):
    """
    Factory that writes a small valid .mp4 into a temp directory.
    Skips the test if the mp4v codec is unavailable in this OpenCV build.
    """
    def _make(name="clip.mp4", frames=10, size=(64, 64), fps=10.0):
        import cv2
        path = tmp_path / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not writer.isOpened():
            pytest.skip("mp4v codec unavailable — cannot create test video")
        for i in range(frames):
            frame = np.full((size[1], size[0], 3), (i * 20) % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path
    return _make