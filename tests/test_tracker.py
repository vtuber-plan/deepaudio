#!/usr/bin/env python
# coding=utf-8
"""
Test Tracker functionality for both Fabric and Accelerate engines.

Usage:
    # Test Fabric tracker
    python tests/test_tracker.py --engine fabric

    # Test Accelerate tracker
    python tests/test_tracker.py --engine accelerate

    # Test both
    python tests/test_tracker.py --engine both
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path

import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from soniq.training.engines import create_engine
from soniq.training.base.context import EngineContext


def test_tracker(engine_type: str, output_dir: str):
    """Test tracker functionality for a specific engine."""
    print(f"\n{'='*60}")
    print(f"Testing {engine_type.upper()} Tracker")
    print(f"{'='*60}\n")

    # Create output directory
    engine_output_dir = Path(output_dir) / engine_type
    engine_output_dir.mkdir(parents=True, exist_ok=True)

    # Create context
    ctx = EngineContext(
        seed=42,
        num_iterations=10,
    )
    ctx.setup_paths(engine_output_dir)

    # Create engine
    engine = create_engine(
        engine_type=engine_type,
        ctx=ctx,
        precision="32-true",
    )

    print(f"Engine: {engine_type}")
    print(f"Output dir: {engine_output_dir}")
    print(f"Metrics path: {ctx.metrics_path}")

    # Test 1: Log scalar metrics
    print("\n[Test 1] Logging scalar metrics...")
    for step in range(5):
        metrics = {
            "train/loss": 1.0 / (step + 1),
            "train/accuracy": 0.5 + 0.1 * step,
            "train/learning_rate": 1e-4 * (0.9 ** step),
        }
        engine.log(metrics, step=step)
        print(f"  Step {step}: logged {list(metrics.keys())}")

    # Test 2: Log audio (if supported)
    print("\n[Test 2] Logging audio...")
    try:
        audio = torch.randn(1, 16000)  # 1 second of random audio at 16kHz
        engine.log_audio("test/audio", audio, sample_rate=16000, step=0)
        print("  Audio logged successfully")
    except Exception as e:
        print(f"  Audio logging failed: {e}")

    # Test 3: Log image (if supported)
    print("\n[Test 3] Logging image...")
    try:
        image = torch.randn(3, 64, 64)  # Random 64x64 RGB image
        engine.log_image("test/image", image, step=0)
        print("  Image logged successfully")
    except Exception as e:
        print(f"  Image logging failed: {e}")

    # Test 4: Check if files were created
    print("\n[Test 4] Checking output files...")
    metrics_path = ctx.metrics_path

    if metrics_path.exists():
        files = list(metrics_path.rglob("*"))
        print(f"  Files in {metrics_path}:")
        for f in files[:10]:  # Show first 10 files
            print(f"    - {f.relative_to(metrics_path)}")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more files")

        # Check for TensorBoard events file
        tfevents = list(metrics_path.rglob("events.out.tfevents.*"))
        if tfevents:
            print(f"\n  ✅ TensorBoard events file found: {tfevents[0].name}")
        else:
            print(f"\n  ⚠️ No TensorBoard events file found")
    else:
        print(f"  ⚠️ Metrics path does not exist: {metrics_path}")

    # Cleanup
    if hasattr(engine, 'end_training'):
        engine.end_training()

    print(f"\n{'='*60}")
    print(f"[SUCCESS] {engine_type.upper()} tracker test completed!")
    print(f"{'='*60}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test Tracker functionality")
    parser.add_argument(
        "--engine",
        type=str,
        default="both",
        choices=["fabric", "accelerate", "both"],
        help="Engine to test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for test artifacts",
    )
    args = parser.parse_args()

    # Create temporary output directory if not specified
    if args.output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="tracker_test_")
        print(f"Using temporary output directory: {output_dir}")
        cleanup = True
    else:
        output_dir = args.output_dir
        cleanup = False

    results = {}

    # Test engines
    if args.engine in ["fabric", "both"]:
        try:
            results["fabric"] = test_tracker("fabric", output_dir)
        except ImportError as e:
            print(f"\n[SKIPPED] Fabric engine not available: {e}")
            results["fabric"] = None
        except Exception as e:
            print(f"\n[FAILED] Fabric engine test failed: {e}")
            import traceback
            traceback.print_exc()
            results["fabric"] = False

    if args.engine in ["accelerate", "both"]:
        try:
            results["accelerate"] = test_tracker("accelerate", output_dir)
        except ImportError as e:
            print(f"\n[SKIPPED] Accelerate engine not available: {e}")
            results["accelerate"] = None
        except Exception as e:
            print(f"\n[FAILED] Accelerate engine test failed: {e}")
            import traceback
            traceback.print_exc()
            results["accelerate"] = False

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for engine, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  {engine.upper()}: {status}")

    # Cleanup
    if cleanup:
        print(f"\nCleaning up temporary directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)

    return 0 if all(p in [True, None] for p in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())