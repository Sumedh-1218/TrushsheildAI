"""
Tests for src/face/face_processor.py
"""

import numpy as np
import pytest

from src.face.face_processor import BaseFaceProcessor, HaarFaceProcessor


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BaseFaceProcessor()


def test_haar_returns_none_on_blank_image():
    """A completely black image has no face, so Haar should return None."""
    proc = HaarFaceProcessor()
    result = proc.extract_face(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result is None


def test_haar_returns_ndarray_or_none():
    """Noise might or might not look like a face, but it shouldn't crash."""
    proc = HaarFaceProcessor()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (120, 120, 3), dtype=np.uint8)
    result = proc.extract_face(img)
    assert result is None or isinstance(result, np.ndarray)