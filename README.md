# Scanner AI - Ultralytics YOLO to OpenVINO

Spectrum channel detector for 2G/3G/4G cellular frequencies. Converts trained Ultralytics YOLO models (.pt) to OpenVINO FP32 and runs inference without PyTorch dependencies.

## Quick Start

### 1. Install dependencies

```bash
pip install ultralytics openvino
```

### 2. Place your trained models

```
2G_MODEL/best.pt
3G_4G_MODEL/best.pt
```

### 3. Convert to OpenVINO

```bash
python export_openvino.py
```

This finds `.pt` files in `2G_MODEL/` and `3G_4G_MODEL/`, converts them to OpenVINO FP32 (always accurate, no quantization), and outputs to `best_openvino_model/` in each directory.

### 4. Run

**Docker (recommended):**
```bash
docker build -t  .
docker run -p 4444:4444 scanner-ai
```

**Direct:**
```bash
pip install -r requirements.txt
python scanner.py
```

## Model Directory Structure

After export:

```
2G_MODEL/
    best.pt                    # trained model (input)
    best_openvino_model/       # converted model (output)
        best.xml
        best.bin
        metadata.yaml

3G_4G_MODEL/
    best.pt
    best_openvino_model/
        best.xml
        best.bin
        metadata.yaml
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCANNER_AI_IP` | `0.0.0.0` | Host to bind to |
| `SCANNER_AI_PORT` | `4444` | TCP port |
| `SAVE_SAMPLES` | `NO` | Save spectrogram images |
| `MEM_OPTIMIZATION` | `YES` | Memory optimization for large bands |
