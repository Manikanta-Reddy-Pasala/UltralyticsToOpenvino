"""
Export trained YOLOv8 model to OpenVINO IR format with optimization.

Supports multiple optimization levels for different NUC hardware:
  --preset speed     INT8 quantization, smallest model, fastest inference
  --preset balanced  INT8 with higher calibration quality
  --preset accuracy  FP16, larger model, best accuracy (needs more RAM)
  --preset debug     FP32, no optimization (for debugging only)

Usage:
    python export_openvino.py
    python export_openvino.py --preset speed
    python export_openvino.py --preset accuracy
    python export_openvino.py --weights runs/detect/train/weights/best.pt
"""

import argparse
import json
import os
import shutil
from datetime import datetime

from ultralytics import YOLO


# =========================================================
# Export Presets
# =========================================================

PRESETS = {
    "speed": {
        "description": "Fastest inference on NUC (~5-10ms). INT8 quantized.",
        "int8": True,
        "half": False,
        "dynamic": False,
        "simplify": True,
        "nms": True,
        "expected_size_mb": "3-4",
        "expected_inference_ms": "5-10",
    },
    "balanced": {
        "description": "Good speed + accuracy. INT8 with careful calibration.",
        "int8": True,
        "half": False,
        "dynamic": False,
        "simplify": True,
        "nms": True,
        "expected_size_mb": "3-5",
        "expected_inference_ms": "8-15",
    },
    "accuracy": {
        "description": "Best accuracy, larger model. FP16 (needs ~100MB more RAM).",
        "int8": False,
        "half": True,
        "dynamic": False,
        "simplify": True,
        "nms": True,
        "expected_size_mb": "6-8",
        "expected_inference_ms": "15-25",
    },
    "debug": {
        "description": "Full FP32, no optimization. For debugging only.",
        "int8": False,
        "half": False,
        "dynamic": False,
        "simplify": False,
        "nms": True,
        "expected_size_mb": "12-14",
        "expected_inference_ms": "25-40",
    },
}


def create_model_metadata(args, preset, model_dir, total_size):
    """Save model metadata for tracking and verification."""
    metadata = {
        "model_name": "spectrum_detector",
        "export_date": datetime.now().isoformat(),
        "source_weights": os.path.basename(args.weights),
        "preset": args.preset,
        "optimization": {
            "int8": preset["int8"],
            "fp16": preset["half"],
            "nms_included": preset["nms"],
        },
        "input": {
            "image_size": args.imgsz,
            "channels": 3,
            "format": "BGR",
        },
        "classes": {
            "0": "2G",
            "1": "3G",
            "2": "4G",
        },
        "num_classes": 3,
        "model_size_mb": round(total_size / 1024 / 1024, 2),
    }

    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to OpenVINO IR")
    parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt",
                        help="Path to trained weights")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (must match training and NUC inference)")
    parser.add_argument("--preset", type=str, default="speed",
                        choices=list(PRESETS.keys()),
                        help="Export preset (speed/balanced/accuracy/debug)")
    parser.add_argument("--int8", action="store_true", default=None,
                        help="Force INT8 quantization (overrides preset)")
    parser.add_argument("--no-int8", action="store_true",
                        help="Force disable INT8 (overrides preset)")
    parser.add_argument("--data", type=str, default="config/dataset.yaml",
                        help="Dataset YAML for INT8 calibration")
    parser.add_argument("--output-dir", type=str, default="./exported_models",
                        help="Copy exported model to this directory")
    args = parser.parse_args()

    preset = PRESETS[args.preset]

    # Allow CLI overrides
    use_int8 = preset["int8"]
    use_half = preset["half"]
    if args.int8 is True:
        use_int8 = True
        use_half = False
    elif args.no_int8:
        use_int8 = False

    print(f"\n{'='*60}")
    print(f" Export Preset: {args.preset}")
    print(f" {preset['description']}")
    print(f"{'='*60}")
    print(f"  Weights:   {args.weights}")
    print(f"  Image:     {args.imgsz}x{args.imgsz}")
    print(f"  INT8:      {use_int8}")
    print(f"  FP16:      {use_half}")
    print(f"  Simplify:  {preset['simplify']}")
    print(f"  NMS:       {preset['nms']}")
    print(f"  Expected:  ~{preset['expected_size_mb']} MB, ~{preset['expected_inference_ms']}ms")
    print(f"{'='*60}\n")

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    print(f"Exporting to OpenVINO IR...")
    export_path = model.export(
        format="openvino",
        imgsz=args.imgsz,
        half=use_half,
        int8=use_int8,
        simplify=preset["simplify"],
        nms=preset["nms"],
        data=args.data if use_int8 else None,
    )

    print(f"\nExported to: {export_path}")

    # Copy to output directory
    dest = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        if os.path.isdir(export_path):
            dest = os.path.join(args.output_dir, "spectrum_detector")
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(export_path, dest)
        else:
            dest = args.output_dir
            shutil.copy2(export_path, dest)

    # Print model size
    total_size = 0
    model_dir = dest or (export_path if os.path.isdir(export_path) else os.path.dirname(export_path))
    print(f"\nModel files:")
    for f in sorted(os.listdir(model_dir)):
        fpath = os.path.join(model_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            print(f"  {f:30s} {size / 1024:8.1f} KB")
    print(f"  {'TOTAL':30s} {total_size / 1024 / 1024:8.2f} MB")

    # Save metadata
    metadata_path = create_model_metadata(args, preset, model_dir, total_size)
    print(f"\n  Metadata saved: {metadata_path}")

    # Comparison table
    print(f"\n{'='*60}")
    print(" Export Complete!")
    print(f"{'='*60}")
    print(f"\n  Model:    {model_dir}")
    print(f"  Size:     {total_size / 1024 / 1024:.2f} MB")
    print(f"  Preset:   {args.preset}")
    print(f"\n  Preset comparison:")
    print(f"  {'Preset':<12} {'Size':<10} {'Speed':<12} {'Accuracy'}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
    for name, p in PRESETS.items():
        marker = " <--" if name == args.preset else ""
        print(f"  {name:<12} {p['expected_size_mb']:>5} MB  {p['expected_inference_ms']:>8}ms   "
              f"{'INT8' if p['int8'] else 'FP16' if p['half'] else 'FP32'}{marker}")

    print(f"\n  Next steps:")
    print(f"  1. Copy the exported model to 2G_MODEL/ or 3G_4G_MODEL/ directory")
    print(f"  2. Run: python scanner.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
