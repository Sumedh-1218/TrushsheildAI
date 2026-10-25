
---

### 📄 File 2: `LEARNING_GUIDE.md`

This file is a deep dive into *why* the code is written the way it is. It's perfect for onboarding new team members or explaining your architecture to judges.

**Where to create it:** Right-click your main `TrustshieldAI` folder ➔ New File ➔ Name it `LEARNING_GUIDE.md`.

Paste this code:

```markdown
# 🧠 TrustShield AI — Learning Guide

This guide explains the architectural decisions behind the Member 3 pipeline. It is intended for future developers who need to understand *how* the system works under the hood.

## 1. The Pipeline Flow
When you call `analyze_video()`, the data flows sequentially:
1. **Validation:** Checks file size, extension, and OpenCV readability.
2. **Metadata:** Extracts FPS, resolution, and duration.
3. **Sampling:** Uses a generator to yield frames at 1-second intervals (configurable). This prevents RAM crashes on long videos.
4. **Analysis:** Each frame goes through Face Extraction ➔ Detector ➔ Output Validation.
5. **Aggregation:** Calculates mean/max scores and identifies top-K evidence frames.
6. **Labeling:** Determines the final risk label based on thresholds.

## 2. Key Design Patterns Used

### A. Dependency Injection
Look at `FrameAnalyzer`. It does not create its own detector or face processor. They are passed into its `__init__`. 
*Why?* Because Member 2 will eventually build a real `TorchDetector`. By using injection, we can swap `MockDetector` for `TorchDetector` without changing a single line of the `FrameAnalyzer` code.

### B. The Abstract Base Class (ABC)
Look at `BaseFaceProcessor` and `BaseDetector`. They use Python's `abc` module. 
*Why?* It forces anyone writing a new model to follow a strict contract. If Member 2 tries to write a detector but forgets the `analyze()` method, Python will throw an error immediately, preventing silent failures.

### C. Generator-Based Frame Sampling
Look at `FrameSampler.sample()`. It uses `yield` instead of `return`. 
*Why?* Video files can be gigabytes. If we loaded all 900 frames into RAM at once, the computer would crash. A generator pulls exactly one frame into memory, analyzes it, and then throws it away before loading the next one. Memory usage stays under 100MB regardless of video length.

### D. The Mock Detector
Look at `MockDetector`. It hashes the image array and converts it to a float between 0.0 and 1.0.
*Why?* We cannot test the pipeline if the real AI model isn't ready. The mock detector provides **deterministic** scores. The same image will always yield the same score, which is essential for writing reliable unit tests.

## 3. The "Inconclusive" Rule
In deepfake detection, you cannot guarantee authenticity. If the Haar Cascade fails to find a face, it returns `None`. 
- The `FrameAnalyzer` catches this and marks the frame as `FRAME_STATUS_UNSCORABLE`.
- If *all* frames are unscorable, the `Aggregator` returns `mean_score=None`.
- The `Pipeline` sees `None` and labels the video as `inconclusive`.
**Never label an unscorable video as "real."**

## 4. How to add the Real AI Model (Member 2's Job)
1. Create `src/detector/real_detector.py`.
2. Make a class `RealDetector(BaseDetector)`.
3. Implement `analyze(image)` to load your PyTorch/TensorFlow model and return a `DetectorOutput`.
4. In `pipeline.py`, change `detector = MockDetector()` to `detector = RealDetector()`.
*That's it. The rest of the pipeline handles the scoring and labeling automatically.*