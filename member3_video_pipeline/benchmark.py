"""
Phase 6: Performance Benchmark
Generates a synthetic video and measures pipeline throughput.
"""

import cv2
import numpy as np
import time
import os
import dataclasses

from src.pipeline import analyze_video
from src.config import DEFAULT_CONFIG

def create_dummy_video(path, num_frames=300, fps=30, size=(640, 480)):
    print(f"🎬 Generating dummy video: {path} ({num_frames} frames at {fps} FPS)...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(num_frames):
        # Create a frame with a moving rectangle so it's not completely static
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.rectangle(frame, (i*5 % size[0], 100), (i*5 % size[0] + 50, 150), (0, 255, 0), -1)
        out.write(frame)
    out.release()
    print("✅ Video generated.")

def run_benchmark():
    video_path = "benchmark_test.mp4"
    
    # 1. Generate a 10-second video (300 frames at 30fps)
    create_dummy_video(video_path, num_frames=300, fps=30)

    # 2. Force the pipeline to sample every frame for a stress test
    # (By default it only samples 1 frame per second)
    test_config = dataclasses.replace(DEFAULT_CONFIG, SAMPLING_INTERVAL_SECONDS=0.033) # ~30fps

    print("\n⏱️  Starting pipeline benchmark (processing every frame)...")
    start_time = time.perf_counter()

    # Run the pipeline
    result = analyze_video(video_path, config=test_config)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # 3. Calculate Stats
    agg = result.get("aggregation", {})
    total_analyzed = agg.get("scored_frames", 0) + agg.get("unscorable_frames", 0) + agg.get("failed_frames", 0)
    fps_processed = total_analyzed / total_time if total_time > 0 else 0

    print("\n" + "="*40)
    print(" 🚀 BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Video Frames : {300}")
    print(f"Frames Analyzed    : {total_analyzed}")
    print(f"Total Time Taken   : {total_time:.4f} seconds")
    print(f"Processing Speed   : {fps_processed:.2f} frames / second")
    print(f"Pipeline Status    : {result.get('status')}")
    print(f"Pipeline Label     : {result.get('label')}")
    print("="*40 + "\n")

    # 4. Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)
        print("🧹 Cleaned up dummy video.")

if __name__ == "__main__":
    run_benchmark()