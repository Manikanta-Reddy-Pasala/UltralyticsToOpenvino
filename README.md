# Scanner AI v2 - OpenVINO Runtime (No PyTorch/Ultralytics)

Spectrum channel detector for 2G/3G/4G cellular frequencies. Runs YOLO models converted to OpenVINO IR format — no PyTorch or Ultralytics dependency at runtime.

## Architecture

```
Spectrogram (float32 FFT data)
  → Normalize power values (-130 to -3 dBm)
  → Apply Viridis colormap → BGR uint8 image
  → 3G/4G detection (YOLOv12n, dynamic shape, OpenVINO)
  → Extract 2G regions (gaps between 3G/4G detections)
  → 2G detection (YOLO11n INT8, dynamic shape, OpenVINO)
  → Convert pixel coordinates → frequency (MHz)
  → Return detected frequencies via protobuf over TCP
```

## Quick Start

### 1. Export models (requires ultralytics, one-time only)

```bash
pip install ultralytics openvino
python export_openvino.py
```

This converts `.pt` files in `2G_MODEL/` and `3G_4G_MODEL/` to OpenVINO FP32 format with `dynamic=True` for flexible input sizes.

For the 2G model, INT8 quantization is recommended for faster inference and lower memory:
```bash
# Export with INT8 (requires calibration data)
yolo export model=2G_MODEL/best.pt format=openvino int8=True imgsz=1216
mv 2G_MODEL/best_int8_openvino_model/ 2G_MODEL/best_int8_openvino_model/
```

### 2. Place models

```
2G_MODEL/
    best_int8_openvino_model/    # INT8 quantized (preferred, faster)
        best.xml
        best.bin
        metadata.yaml
    best_openvino_model/         # FP32 fallback
        best.xml
        best.bin
        metadata.yaml

3G_4G_MODEL/
    best_openvino_model/         # FP32 (dynamic shape)
        best.xml
        best.bin
        metadata.yaml
```

### 3. Run

**Docker (recommended):**
```bash
docker build -t scanner-ai .
docker run -p 4444:4444 -e MEM_OPTIMIZATION=YES scanner-ai
```

**Direct:**
```bash
pip install -r requirements.txt
MEM_OPTIMIZATION=YES python scanner.py
```

## Runtime Dependencies

- `openvino >= 2024.0.0` — inference engine
- `opencv-python-headless >= 4.8.0` — image preprocessing
- `numpy >= 1.24.0` — array operations
- `protobuf == 3.20` — TCP message serialization
- `Pillow` — image utilities

No PyTorch, no Ultralytics, no CUDA required at runtime.

## Models

| Model | Architecture | Input Shape | Classes | Quantization |
|-------|-------------|-------------|---------|--------------|
| 2G | YOLO11n | Dynamic (stride 32) | 2G (GSM) | INT8 (recommended) or FP32 |
| 3G/4G | YOLOv12n | Dynamic (stride 32) | 3G (UMTS), 4G (LTE), 4G-TDD | FP32 |

Both models are exported with `dynamic=True`, which traces symbolic dimensions through the ONNX graph (including attention layers in YOLOv12n). This allows stride-aligned minimal padding (`auto=True`) at runtime, preserving full detection accuracy across all band widths without the resolution loss of static 640x640 letterboxing.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCANNER_AI_IP` | `0.0.0.0` | Host to bind to |
| `SCANNER_AI_PORT` | `4444` | TCP port |
| `SAVE_SAMPLES` | `NO` | Save spectrogram images to `SAMPLES_LOW_POWER/` |
| `MEM_OPTIMIZATION` | `YES` | Split large bands (>100 MHz) into chunks |

## Protocol

TCP socket on port 4444 using protobuf messages:

1. Client → `AIPredictSampleReq` (band parameters: center freq, bandwidth, chunks)
2. Server ← `AIPredictSampleRes` (acknowledgment)
3. Client → `AISampleDataReq` (float32 spectrum data)
4. Server ← `AISampleDataRes` (detected frequencies: `lte_freqs`, `umts_freqs`, `gsm_freqs`)

## Testing

Start the scanner, then run pytest:

```bash
MEM_OPTIMIZATION=YES python scanner.py &
sleep 20  # wait for model warmup
pytest testing/test_scanner_ai_script.py -v
```

Test samples are in `SAMPLES_UT/` (6 bands: B1, B3, B8, B20, B28, B40).

## Performance

- Shared OpenVINO Core instance across models
- 2G model uses INT8 quantization for faster inference
- Singleton colormap/normalizer (no per-call allocation)
- Direct float32 blob construction (skips uint8→float32 conversion)
- Memory optimization splits large bands (>100 MHz) into 60 MHz chunks with 12 MHz overlap
- Thread-safe: each connection uses local result lists

| Band | Bandwidth | Inference Time |
|------|-----------|---------------|
| B20 | 30 MHz | ~140 ms |
| B28 | 50 MHz | ~285 ms |
| B1 | 60 MHz | ~360 ms |
| B3 | 80 MHz | ~360 ms |
| B8 | 40 MHz | ~330 ms |
| B40 | 100 MHz | ~1.9 s |
