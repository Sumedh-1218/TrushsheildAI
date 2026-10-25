"""
TrustShield AI — Member 3 — Face Processor
src/face/face_processor.py

Abstract face processing interface + temporary Haar cascade implementation.
"""

from __future__ import annotations

import abc
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import PipelineConfig, DEFAULT_CONFIG


# ─── Abstract Interface ───────────────────────────────────────────────────────

class BaseFaceProcessor(abc.ABC):
    @abc.abstractmethod
    def extract_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def processor_name(self) -> str:
        raise NotImplementedError


# ─── Temporary Haar Cascade Implementation ────────────────────────────────────

class HaarFaceProcessor(BaseFaceProcessor):
    PROCESSOR_NAME = "HaarFaceProcessor_v1_TEMPORARY"

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config
        self._cascade = self._load_cascade()

    def _load_cascade(self) -> Optional[cv2.CascadeClassifier]:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            import warnings
            warnings.warn(
                "HaarFaceProcessor: Could not load Haar cascade. "
                "Face detection will not function.",
                RuntimeWarning,
            )
            return None
        return cascade

    def extract_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self._cascade is None or image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.FACE_SCALE_FACTOR,
            minNeighbors=self.config.FACE_MIN_NEIGHBORS,
            minSize=(self.config.MIN_FACE_SIZE, self.config.MIN_FACE_SIZE),
        )

        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        h_img, w_img = image.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        return face_crop

    @property
    def processor_name(self) -> str:
        return self.PROCESSOR_NAME


# ─── Full-Frame fallback ──────────────────────────────────────────────────────

class FullFrameProcessor(BaseFaceProcessor):
    PROCESSOR_NAME = "FullFrameProcessor_v1"

    def extract_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        return image if image is not None and image.size > 0 else None

    @property
    def processor_name(self) -> str:
        return self.PROCESSOR_NAME


# ─── Factory ─────────────────────────────────────────────────────────────────

def get_default_face_processor(config: PipelineConfig = DEFAULT_CONFIG) -> BaseFaceProcessor:
    return HaarFaceProcessor(config)