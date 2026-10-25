# TrustShield AI — Deepfake Video Analysis Pipeline

This repository contains the core analytical pipeline for **TrustShield AI**, designed to detect deepfake indicators in video files. 

The pipeline handles video validation, frame sampling, face extraction, model inference, and evidence aggregation. It is designed with a **Responsible AI** first approach, ensuring that inconclusive results are explicitly labeled as such, rather than defaulting to "real."

## 🚀 Features
- **Fail-Safe Validation:** Rejects oversized, corrupt, or unsupported files before processing.
- **Efficient Sampling:** Extracts frames at configurable intervals (default: 1 fps) to optimize processing speed.
- **Modular Architecture:** Face processors and deepfake detectors are injected dependencies, making it easy to swap in the real model (Member 2) later.
- **Deterministic Mocking:** Includes a `MockDetector` for testing the pipeline end-to-end without needing the actual AI model.
- **JSON-Ready Output:** Returns a clean dictionary that is 100% JSON serializable for frontend/API integration.

## 📂 Repository Structure
```text
src/
├── analysis/       # Frame orchestration, aggregation, and evidence generation
├── detector/       # Detector interfaces and MockDetector
├── face/           # Face extraction (Haar Cascade temporary impl)
├── schemas/        # Data contracts (Dataclasses)
├── validation/     # Pre-flight video checks
├── video/          # OpenCV metadata reading and frame sampling
└── pipeline.py     # Main public API and CLI entry point
tests/
├── ...             # 69 unit and integration tests