# Scanner AI v3 - Ultralytics YOLO to OpenVINO

Spectrum channel detector for 2G/3G/4G cellular frequencies using YOLO models exported to OpenVINO IR format. Runs inference without PyTorch or Ultralytics dependencies - pure OpenVINO runtime.

## Prerequisites

- Python >= 3.10
- OpenVINO model files (not included in this repo)

## Models Required (Not Included)

You must place your exported OpenVINO models in the following directories:

```
2G_MODEL/best_int8_openvino_model/
    best.xml
    best.bin
    metadata.yaml

3G_4G_MODEL/best_openvino_model/
    best.xml
    best.bin
    metadata.yaml
```

### How to Export Models from Ultralytics

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Export to OpenVINO (FP32)
model.export(format="openvino", imgsz=[640, 640])

# Export to OpenVINO (INT8 quantized - smaller & faster)
model.export(format="openvino", imgsz=[1216, 3360], int8=True)
```

## How to Run

### Option 1: Run Directly

```bash
# Install dependencies
pip install -r requirements.txt

# Run the scanner service (listens on port 4444)
python scanner.py
```

### Option 2: Run with Docker

```bash
# Build the Docker image
docker build -t scanner-ai-v3 .

# Run the container
docker run -p 4444:4444 \
  -e SCANNER_AI_PORT=4444 \
  -e SAVE_SAMPLES=NO \
  -e MEM_OPTIMIZATION=YES \
  scanner-ai-v3
```

### Option 3: Run with Docker Compose

```bash
# Set required environment variables
export SCANNER_AI_VERSION=latest
export SCANNER_AI_LOW_POWER_SAMPLES=/path/to/samples

# Start
docker compose up -d
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCANNER_AI_IP` | `0.0.0.0` | Host to bind to |
| `SCANNER_AI_PORT` | `4444` | TCP port to listen on |
| `SAVE_SAMPLES` | `NO` | Save spectrogram images (`YES`/`NO`) |
| `MEM_OPTIMIZATION` | `YES` | Enable memory optimization for large bands (`YES`/`NO`) |

## How to Test

The test suite connects to a running scanner service via TCP socket and sends spectrum sample data.

### Prerequisites for Testing

1. The scanner service must be running on `127.0.0.1:4444`
2. Test sample data files must be placed in `SAMPLES_UT/` directory (not included in repo):
   - `sample_vec_B1.dat`
   - `sample_vec_B3.dat`
   - `sample_vec_B8.dat`
   - `sample_vec_B20.dat`
   - `sample_vec_B28.dat`
   - `sample_vec_B40.dat`

### Run Tests

```bash
# Install test dependencies
pip install pytest coverage

# Start the scanner service first (in a separate terminal)
python scanner.py

# Run tests
python -m pytest testing/ -v

# Run tests with coverage
coverage run -m pytest testing/ -v
coverage report
```

### Expected Test Results

The test validates detected frequencies per band:

| Band | Expected 4G (MHz) | Expected 3G (MHz) | Expected 2G (MHz) |
|------|--------------------|--------------------|--------------------|
| B1   | 2165.0, 2146.7     | 2116.4, 2137.7     | -                  |
| B3   | 1815.0, 1870.0, 1849.5 | -              | 1860.2             |
| B8   | -                  | 932.6, 937.2, 927.5 | 953.4            |
| B20  | 813.6, 798.5       | -                  | -                  |
| B28  | 763.1, 800.8       | -                  | -                  |
| B40  | 2342.1, 2361.9     | -                  | -                  |

## Architecture

```
Scanner AI Service (TCP :4444)
    |
    |-- Receives protobuf messages (AIPredictSampleReq + AISampleDataReq)
    |-- Builds spectrogram from raw FFT samples
    |-- Applies viridis colormap
    |
    |-- 3G/4G Detection (YOLOv12n, OpenVINO FP32, 640x640)
    |       Classes: 3G, 4G, 4G-TDD
    |       Confidence threshold: 0.6
    |
    |-- 2G Detection (YOLO11n, OpenVINO INT8, 1216x3360)
    |       Classes: 2G
    |       Confidence threshold: 0.3
    |
    |-- Returns detected frequencies via protobuf (AISampleDataRes)
```

## File Structure

```
.
├── scanner.py              # Main inference service
├── ai_colormap.py          # Viridis colormap & normalization
├── viridis_colormap.py     # Viridis color lookup table
├── ai_model_pb2.py         # Protobuf generated code
├── scanner_logging.py      # Logging setup
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project config (uv compatible)
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker Compose config
├── .python-version         # Python version pin (3.10)
├── testing/
│   └── test_scanner_ai_script.py  # Integration tests
├── 2G_MODEL/               # (not in repo) 2G OpenVINO model
│   └── best_int8_openvino_model/
├── 3G_4G_MODEL/            # (not in repo) 3G/4G OpenVINO model
│   └── best_openvino_model/
└── SAMPLES_UT/             # (not in repo) Test sample data
```
