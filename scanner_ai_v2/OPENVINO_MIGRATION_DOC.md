# Scanner AI v2: Technical Reference and OpenVINO Migration Guide

**Project**: Scanner AI v2 - Wideband RF Signal Detection with AI Inference
**Version**: 2.0.0
**Repository**: https://github.com/Manikanta-Reddy-Pasala/UltralyticsToOpenvino
**Last Updated**: 2026-02-21
**Migration Commit**: `3281651` - `fix(scanner-ai): migrate inference to OpenVINO and fix model compatibility`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Project Structure](#4-project-structure)
5. [Signal Processing Pipeline](#5-signal-processing-pipeline)
6. [OpenVINO Inference Engine](#6-openvino-inference-engine)
7. [AI Models](#7-ai-models)
8. [TCP Server and Protobuf Protocol](#8-tcp-server-and-protobuf-protocol)
9. [Colormap and Power Normalization](#9-colormap-and-power-normalization)
10. [Migration: Ultralytics to OpenVINO](#10-migration-ultralytics-to-openvino)
11. [Migration Advantages and Known Issues](#11-migration-advantages-and-known-issues)
12. [Model Export Process](#12-model-export-process)
13. [Docker and Deployment](#13-docker-and-deployment)
14. [Configuration and Environment Variables](#14-configuration-and-environment-variables)
15. [Testing](#15-testing)
16. [Performance Characteristics](#16-performance-characteristics)
17. [Memory Management](#17-memory-management)
18. [Dependency Analysis](#18-dependency-analysis)
19. [Design Decisions and Rationale](#19-design-decisions-and-rationale)
20. [Troubleshooting Guide](#20-troubleshooting-guide)
21. [Appendices](#21-appendices)

---

## 1. Executive Summary

Scanner AI v2 is a real-time wireless signal detection service that identifies 2G (GSM), 3G (UMTS), and 4G (LTE/LTE-TDD) cellular signals embedded within wideband RF spectrum data. The system receives raw FFT (Fast Fourier Transform) power spectrogram data from an external radio scanner device, converts it into visual spectrogram images, and applies trained YOLO object detection models to locate and classify signals by their spectral signatures.

The service runs as a persistent TCP socket server on port 4444, accepting protobuf-encoded requests and returning detected center frequencies in MHz for each signal technology.

The defining architectural change introduced in v2 is the migration from the Ultralytics runtime (which requires PyTorch) to Intel's OpenVINO inference runtime. This migration reduces the deployment footprint by approximately 800 MB to 1 GB, eliminates Python runtime dependencies on PyTorch and CUDA, and improves inference determinism by requiring explicit custom implementations of preprocessing, postprocessing, and Non-Maximum Suppression. Models are exported once from `.pt` (PyTorch) format to OpenVINO IR format (`.xml` + `.bin`) using a separate export script and are then used purely with the lightweight OpenVINO runtime in production.

**Key facts at a glance:**

| Attribute | Value |
|-----------|-------|
| Language | Python 3.10+ |
| Runtime inference | OpenVINO >= 2024.0.0 |
| Models | YOLO11n (2G), YOLOv12n (3G/4G) |
| Transport protocol | TCP sockets with protobuf framing |
| Service port | 4444 |
| Inference latency | 237–554 ms per band (Apple M2) |
| Container image base | Ubuntu 22.04 (multi-stage) |
| Package manager | `uv` (Astral) |
| Docker runtime dependencies | openvino, numpy, opencv-python-headless, protobuf |

---

## 2. System Overview

### 2.1 Problem Domain

Modern wireless network testing requires identifying which cellular technologies (2G, 3G, 4G) are operating at which frequencies within a measured spectrum. Traditional approaches scan sequentially through candidate frequencies and attempt decoding at each step - an expensive, slow process. Scanner AI v2 shortens this by using visual AI: a spectrogram image of the entire band is analyzed by object detection models trained to recognize the distinctive spectral "shape" of each technology's signal, returning candidate center frequencies in a single inference pass.

### 2.2 Inputs and Outputs

**Input**: Raw float32 FFT power values from a wideband RF scanner, delivered as a flat array over TCP. The scanner sends metadata (center frequency, bandwidth, number of frequency chunks, overlap) followed by the sample data encoded as a protobuf message.

**Output**: Three lists of center frequencies in MHz:
- `lte_freqs` - 4G (LTE FDD and LTE TDD) center frequencies
- `umts_freqs` - 3G (UMTS/WCDMA) center frequencies
- `gsm_freqs` - 2G (GSM) center frequencies

### 2.3 Inference Strategy: Two-Stage Detection

The system uses a deliberate two-stage approach:

**Stage 1 - 3G/4G Detection (YOLOv12n)**
The full spectrogram image is fed to the 3G/4G model. This model detects 3G and 4G signals by class (`cls=0` for 3G, `cls=1` for 4G FDD, `cls=2` for 4G TDD). The bounding box x-coordinates of detections are recorded as "occupied" spectrum segments.

**Stage 2 - 2G Detection (YOLO11n)**
The image is masked by removing the segments identified in Stage 1. The remaining image (gaps between 3G/4G signals) is assembled into a new composite image and fed to the 2G model. This works because 2G (GSM) signals spectrally appear within or adjacent to the 3G/4G bands and their narrower 200 kHz channels can be masked by the coarser 5 MHz / 10 MHz detections of 3G/4G models if both were run simultaneously.

### 2.4 Spectrogram Representation

The scanner hardware captures spectrum data as a 2D array where:
- **Rows** = time snapshots (sweeps)
- **Columns** = frequency bins (2048 bins per sweep, 15 kHz/bin spacing)

Only frequency bins `[357:1691]` (1334 columns wide) are retained from the full 2048-bin FFT, corresponding to the usable in-band portion of the spectrum after filtering guard bands. The retained band represents approximately 20 MHz of spectrum per center frequency sweep.

---

## 3. Architecture Diagram

### 3.1 High-Level System Context

```
  RF Scanner Hardware
        |
        | (Wideband RF capture)
        v
  Scanner Device Software
        |
        | TCP connection to port 4444
        | Protobuf: AIModelReq (predict_sample_req + sample_data_req)
        v
 +-------------------------------+
 |     Scanner AI v2 Service     |
 |   (scanner.py, port 4444)     |
 |                               |
 |  [TCP Thread per connection]  |
 |           |                   |
 |  [Signal Processing Pipeline] |
 |           |                   |
 |  [OpenVINO 3G/4G Inference]   |
 |           |                   |
 |  [2G Image Masking]           |
 |           |                   |
 |  [OpenVINO 2G Inference]      |
 |           |                   |
 |  [Freq Coordinate Transform]  |
 |                               |
 +-------------------------------+
        |
        | Protobuf: AIModelRes (sample_data_res)
        | lte_freqs, umts_freqs, gsm_freqs
        v
  Scanner Device Software
```

### 3.2 Internal Inference Pipeline

```
  TCP recv (header protobuf: AIModelReq.predict_sample_req)
        |
        | Extract: center_freq, bandwidth, num_chunks, overlap, sample_len
        v
  ACK sent: AIModelRes.predict_sample_res (SUCCESS)
        |
  TCP recv (data protobuf: AIModelReq.sample_data_req)
        |
        | samples: float32 array (length = rows * 2048)
        v
  np.reshape(length // 2048, 2048)      <-- [rows, 2048]
        |
  spectrogram = data[:, 357:1691]       <-- trim to [rows, 1334]
        |
  create_correct_spectrogram_by_rearranging_samples()
        |                               <-- stitch multi-center-freq bands
        v
  [If bandwidth >= 100 MHz: split into 60 MHz chunks with 12 MHz overlap]
        |
  For each chunk:
        |
        |----> NormalizePowerValue.get_normalized_values()
        |         clip[-130, -3 dBm] -> quantize 0.5 dBm -> index [0, 254]
        |
        |----> CustomImg.get_new_img()
        |         index -> viridis RGB lookup -> BGR uint8 image
        |
        |----> run_inference(3G/4G model)
        |         preprocess_image()  [letterbox -> float32 blob (1,3,H,W)]
        |         compiled_model([blob])[output_layer]  [OpenVINO infer]
        |         postprocess_yolo_output()  [transpose, filter, NMS]
        |
        |----> process_results_3g_4g()
        |         pixel cx -> freq (MHz), record x-segments to mask
        |
        |----> Build 2G composite image (gaps between 3G/4G segments)
        |
        |----> run_inference(2G model)
        |         same preprocess -> infer -> postprocess pipeline
        |
        |----> process_results_2g()
        |         pixel cx -> freq (MHz) via chunk-aware coordinate map
        |
  Aggregate: predicted_freq_list_4g, predicted_freq_list_3g, predicted_freq_list_2g
        |
  AIModelRes.sample_data_res -> TCP send
```

### 3.3 OpenVINO Model Loading

```
  Startup (module-level, before TCP server starts)
        |
  _ov_core = ov.Core()                         <-- single shared Core
  set_property("CPU", {LATENCY, NUM_THREADS})
        |
  load_openvino_model("2G_MODEL/best_openvino_model/")
        |--- read .xml -> compile for CPU
        |--- parse metadata.yaml for imgsz (if dynamic shape)
        |--- return: (compiled_model, output_layer, input_shape)
        |
  load_openvino_model("3G_4G_MODEL/best_openvino_model/")
        |--- same process
        |
  Model warmup (dummy inference to JIT-compile kernels)
        |
  TCP server starts (s.listen(1))
```

---

## 4. Project Structure

```
scanner_ai_v2/
|
|-- scanner.py                    # Main entrypoint (626 lines)
|   |                             # TCP server, OpenVINO engine, full pipeline
|   |-- _ov_core                  # Singleton ov.Core instance
|   |-- load_openvino_model()     # Model loader
|   |-- preprocess_image()        # Letterbox + normalization
|   |-- numpy_nms()               # Pure NumPy NMS
|   |-- postprocess_yolo_output() # Tensor -> detections
|   |-- run_inference()           # Full inference pipeline
|   |-- predict_samples()         # Per-request processing orchestrator
|   |-- recieve_samples()         # TCP connection handler
|   `-- main()                    # Server startup
|
|-- export_openvino.py            # One-time model export script
|   |                             # Requires ultralytics (not in production deps)
|   `-- Uses model.export(format="openvino", dynamic=True)
|
|-- ai_colormap.py                # Power normalization + colormap application
|   |-- NormalizePowerValue       # clip/quantize dBm values -> indices
|   `-- CustomImg                 # Lookup table application (viridis)
|
|-- viridis_colormap.py           # Pre-computed viridis RGB table (256 x 3)
|                                 # Avoids matplotlib dependency at runtime
|
|-- scanner_logging.py            # Logging configuration (stdout, timestamped)
|
|-- ai_model_pb2.py               # Auto-generated protobuf classes
|                                 # AIModelReq, AIModelRes, AIResult, etc.
|
|-- pyproject.toml                # Project metadata + runtime dependencies
|-- requirements.txt              # Flat pip requirements (same deps)
|-- uv.lock                       # Locked dependency tree for uv
|-- .python-version               # Python version pin for uv
|
|-- Dockerfile                    # Multi-stage Ubuntu 22.04 build
|-- Dockerfile.python             # Alternative Python-base Dockerfile
|-- docker-compose.yml            # Service definition for deployment
|
|-- 2G_MODEL/
|   |-- best.pt                   # PyTorch source (50 MB, not needed at runtime)
|   |-- best.onnx                 # ONNX intermediate (not used in production)
|   |-- best_openvino_model/      # FP32 OpenVINO IR (production, ~99 MB dir)
|   |   |-- best.xml              # Model graph (topology)
|   |   |-- best.bin              # Model weights (binary)
|   |   |-- metadata.yaml         # imgsz, classes, export config
|   |   `-- model_metadata.json   # Export provenance record
|   `-- best_int8_openvino_model/ # INT8 quantized (fallback, not preferred)
|
|-- 3G_4G_MODEL/
|   |-- best.pt                   # PyTorch source (5.3 MB, not needed at runtime)
|   |-- best_openvino_model/      # FP32 OpenVINO IR (production, ~21 MB dir)
|   |   |-- best.xml
|   |   |-- best.bin
|   |   |-- metadata.yaml
|   |   `-- model_metadata.json
|   `-- best_openvino_model/      # (nested - see note in Appendix A)
|
|-- testing/
|   `-- test_scanner_ai_script.py # Integration test suite (6 bands)
|
`-- SAMPLES_UT/                   # Binary test spectrogram files
    |-- sample_vec_B1.dat
    |-- sample_vec_B3.dat
    |-- sample_vec_B8.dat
    |-- sample_vec_B20.dat
    |-- sample_vec_B28.dat
    `-- sample_vec_B40.dat
```

---

## 5. Signal Processing Pipeline

### 5.1 Raw Data Reception

The scanner hardware sends FFT power measurements as a flat array of `float32` values. Each value is a power measurement in dBm at a specific frequency bin and time instant.

```
Flat float32 array:
[p(t=0,f=0), p(t=0,f=1), ..., p(t=0,f=2047),
 p(t=1,f=0), p(t=1,f=1), ..., p(t=1,f=2047),
 ...
 p(t=N,f=0), p(t=N,f=1), ..., p(t=N,f=2047)]
```

This is reshaped to a 2D spectrogram matrix:

```python
sample = sample.reshape(length // fft_size, fft_size)
# Result: shape [rows, 2048]
# rows = number of time sweeps
# 2048 = FFT bins per sweep
# Each bin = 15 kHz of spectrum
```

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`, lines 573-576

### 5.2 Frequency Bin Trimming

The full 2048-bin FFT contains guard bands and filter roll-off at the edges. Only bins 357 through 1690 (1334 bins) carry reliable signal information:

```python
spectrogram = data[:, 357:1691]
# Result: shape [rows, 1334]
# 1334 bins * 15 kHz/bin = 20.01 MHz per center frequency
```

This trim removes the outer ~26.75% of FFT bins on each side. The usable 1334-bin window represents the flat portion of the digital filter's passband.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`, line 434

### 5.3 Multi-Center-Frequency Stitching

When the target band is wider than what a single scanner sweep can capture (~20 MHz), the hardware performs multiple overlapping sweeps at different center frequencies (`num_center_frequencies`). The raw array interleaves these sweeps and they must be stitched together by `create_correct_spectrogram_by_rearranging_samples()`.

**Parameters controlling stitching:**
- `num_center_frequencies`: Number of center frequencies used (from request metadata)
- `num_of_samples_in_freq`: Total rows divided by num center frequencies (rows per center freq)
- `fifteen_mhz_points = 1000`: Number of bins covering 15 MHz (1000 * 15 kHz = 15 MHz)
- `five_mhz_points = 333`: Number of bins covering 5 MHz (333 * 15 kHz = 5 MHz)

The stitching logic joins the 15 MHz left portion of the first sweep with the 5 MHz middle portions of subsequent sweeps, and the full tail of the last sweep. This removes the 5 MHz overlap at each boundary.

```
Sweep 0: [---15 MHz---][5MHz][5MHz]
Sweep 1:          [5MHz][---10 MHz---][5MHz]
Sweep 2:                     [5MHz][---15 MHz---]

Stitched: [---15 MHz---][---10 MHz---][---15 MHz---] = 40 MHz
```

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`, lines 353-385

### 5.4 Memory Optimization Chunking

For bands wider than 100 MHz, the stitched spectrogram would produce an extremely wide image that exceeds practical inference dimensions. The `get_num_chunks_for_mem_optimization()` function splits wide bands into 60 MHz chunks with 12 MHz overlap:

```python
MAX_AI_BANDWIDTH = 60 * 1e3   # 60 MHz (in kHz units)
overlap = 12 MHz / 15 kHz = 800 bins

num_chunks = int(bandwidth // (60 MHz - overlap_in_kHz))
```

**Rationale**: The YOLO models were trained on spectrogram images representing approximately 60 MHz of bandwidth. Feeding a 100 MHz wide image to a model trained on 60 MHz images causes severe accuracy degradation because signal aspect ratios become incorrect. Chunking preserves the expected frequency-to-pixel density that the model learned during training.

The 12 MHz overlap between chunks prevents missing signals that straddle a chunk boundary. A signal detected in the overlap zone will appear in both chunks; deduplication occurs naturally because the frequency coordinate rounding produces identical MHz values.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`, lines 345-350, 448-456

### 5.5 Power Normalization

The `NormalizePowerValue` class in `ai_colormap.py` converts raw dBm values into colormap indices:

```python
# Step 1: Clip to expected dynamic range
img = np.clip(img, -130, -3)       # Ignore values outside [-130, -3] dBm

# Step 2: Quantize to 0.5 dBm steps
img = np.round(img / 0.5) * 0.5

# Step 3: Convert to index in [0, 254]
img = np.abs((img - (-130)) / 0.5)
# -130 dBm -> index 0  (noise floor, will map to dark viridis color)
# -3   dBm -> index 254 (strong signal, will map to bright viridis color)
```

**Design rationale**: The 0.5 dBm quantization step and 127 dB dynamic range [-130, -3] were chosen to match the training data preparation. The models were trained on spectrogram images generated with exactly these normalization parameters. Using a different normalization would cause inference accuracy to degrade even if the model architecture is unchanged.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/ai_colormap.py`, lines 26-35

### 5.6 Viridis Colormap Application

The normalized power indices [0, 254] are mapped to RGB colors using the Viridis perceptual colormap, stored as a pre-computed 256x3 NumPy array in `viridis_colormap.py`:

```python
# viridis_colormap.py contains:
var = np.array([
    [68, 1, 84],    # index 0: dark purple (noise floor)
    [69, 2, 86],    # index 1: ...
    ...
    [253, 231, 37], # index 254: bright yellow (strong signal)
])

# Application (in ai_colormap.py):
def get_new_img(self, img):
    return self.colors[img.astype(int)]  # NumPy fancy indexing
```

The result is a [rows, columns, 3] RGB uint8 image array. This is then reversed to BGR channel order (for OpenCV compatibility):

```python
colormapped_array = colormapped_array[..., ::-1]  # RGB -> BGR
```

**Why Viridis**: Viridis is a perceptually uniform colormap designed to have consistent luminance gradients across its range. This means that signal boundaries appear as sharp transitions in both hue and luminance, making them easier for the convolutional neural network to detect as edges. The pre-computed lookup table avoids any matplotlib dependency at runtime.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/ai_colormap.py`, `/Users/manip/Documents/codeRepo/scanner_ai_v2/viridis_colormap.py`

### 5.7 Frequency Coordinate Conversion

After inference, bounding box center x-coordinates (in pixels) are converted to frequencies in MHz.

**For 3G/4G detections** (regular grid, `process_results_3g_4g`):

```python
predicted_center_freq = ((bandwidth / pixel_size_of_img) * xval)
predicted_center_freq += start_freq
predicted_center_freq /= 1e6   # Hz -> MHz
predicted_center_freq = round(predicted_center_freq, 1)
```

Where:
- `bandwidth` = total bandwidth in kHz
- `pixel_size_of_img` = image width in pixels
- `xval` = detection center x in pixels
- `start_freq` = `center_freq - bandwidth/2`

**For 2G detections** (chunk-aware coordinate map, `process_results_2g`):

The 2G inference image is a composite of non-contiguous spectrum segments (the gaps between 3G/4G detections). The `chunk_start_indexes_in_new_image` list records the x-pixel offset where each gap segment begins in the composite image. The `start_freq_for_each_chunk` list records the corresponding starting frequency:

```python
# Determine which segment the detection falls in
while i < len(chunk_start_indexes_in_new_image):
    if x_val_in_image < chunk_start_indexes_in_new_image[i]:
        index_val = i - 1
    else:
        index_val = i
    i += 1

# Convert pixel within segment to frequency
predicted_center_freq = (15000 * (x_val_in_image - chunk_start_indexes_in_new_image[index_val]))
predicted_center_freq += start_freq_for_each_chunk[index_val]
predicted_center_freq /= 1e6
```

The constant `15000` is the Hz/pixel ratio: each pixel in the original spectrogram represents 15 kHz.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`, lines 292-342

---

## 6. OpenVINO Inference Engine

### 6.1 Shared Core Instance

A single `ov.Core()` instance is created at module load time (before the TCP server starts):

```python
# scanner.py, lines 62-66
_ov_core = ov.Core()
_ov_core.set_property("CPU", {
    "INFERENCE_NUM_THREADS": str(os.cpu_count() or 4),
    "PERFORMANCE_HINT": "LATENCY",
})
```

**Why a shared Core**: Creating multiple `ov.Core()` instances multiplies initialization overhead and memory usage for the OpenVINO plugin system. A single shared instance allows the runtime to maintain one plugin per device (CPU in this case) and share optimization state across models.

**LATENCY hint**: OpenVINO's `PERFORMANCE_HINT: LATENCY` instructs the runtime to optimize for single-request latency rather than throughput. This is correct for Scanner AI because each TCP request is processed sequentially (one band at a time) and minimizing per-request latency is the priority over batching throughput.

**Thread count**: Setting `INFERENCE_NUM_THREADS` to `os.cpu_count()` allows OpenVINO to use all available CPU cores for parallel execution of model layers. On a deployment host with 4–8 cores this significantly reduces inference time.

### 6.2 Model Loading

```python
def load_openvino_model(model_dir):
    # 1. Find .xml file (model topology)
    xml_files = [f for f in os.listdir(model_dir) if f.endswith(".xml")]
    model_file = os.path.join(model_dir, xml_files[0])

    # 2. Read and compile
    model = _ov_core.read_model(model_file)
    compiled_model = _ov_core.compile_model(model, "CPU")

    # 3. Get output layer reference
    output_layer = compiled_model.output(0)

    # 4. Determine input shape (handle dynamic shapes)
    partial_shape = compiled_model.input(0).get_partial_shape()
    if partial_shape.is_static:
        input_shape = list(partial_shape.to_shape())
    else:
        # Dynamic shape: read imgsz from metadata.yaml
        meta_imgsz = _read_model_imgsz(model_dir)
        input_shape = [1, 3, meta_imgsz[0], meta_imgsz[1]]

    return compiled_model, output_layer, input_shape
```

The `.xml` file is the model graph (topology) in OpenVINO's Intermediate Representation (IR) format. The accompanying `.bin` file contains the raw weight data. OpenVINO reads both automatically given only the `.xml` path.

**Dynamic shape handling**: Models exported with `dynamic=True` have dynamic input dimensions. The `is_static` check detects this and falls back to reading the intended image size from `metadata.yaml`. The metadata parser (`_read_model_imgsz`) uses only the standard library to avoid a PyYAML dependency.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`, lines 34-90

### 6.3 Image Preprocessing (Letterbox)

```python
def preprocess_image(img, target_h, target_w, auto=False, stride=32):
    orig_h, orig_w = img.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    if auto:
        # Stride-aligned padding (for dynamic shape models)
        pad_w = (stride - new_w % stride) % stride
        pad_h = (stride - new_h % stride) % stride
        canvas_w = new_w + pad_w
        canvas_h = new_h + pad_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
    else:
        # Fixed canvas (for static shape models)
        canvas_w = target_w
        canvas_h = target_h
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Build float32 blob directly (avoids intermediate uint8 canvas)
    blob = np.full((canvas_h, canvas_w, 3), 114.0 / 255.0, dtype=np.float32)
    blob[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized.astype(np.float32) * (1.0 / 255.0)

    # HWC -> CHW, BGR -> RGB, add batch dim
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[::-1])[np.newaxis]

    letterbox_info = {'scale': scale, 'pad_w': pad_left, 'pad_h': pad_top,
                      'orig_w': orig_w, 'orig_h': orig_h}
    return blob, letterbox_info
```

The `letterbox_info` dictionary carries the geometric transformation parameters needed to map detection coordinates back from the padded blob space to the original image space.

**`auto=True` mode** (stride-aligned padding): Used for models exported with `dynamic=True`. Instead of padding to a fixed target size, the canvas is padded to the nearest multiple of the YOLO stride (32 pixels). This produces the smallest valid input tensor for a given image, reducing compute time proportionally.

**`auto=False` mode** (fixed canvas): Used only if a model has a static input shape. The padded canvas always matches the fixed shape the model was compiled for.

**Performance optimization - direct float32 blob**: The blob is built directly in `float32` rather than creating a `uint8` canvas and converting. This avoids an intermediate array allocation and type conversion. The fill value `114.0/255.0 ≈ 0.447` is the standard YOLO letterbox gray value normalized to float.

**Channel reversal `[::-1]`**: The transpose makes CHW layout; the `[::-1]` on the first axis (channel axis after transpose) reverses BGR to RGB. This combined operation is a zero-copy view and requires `np.ascontiguousarray` to materialize as a contiguous array before passing to OpenVINO.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`, lines 93-129

### 6.4 Raw Tensor Output Structure

YOLO11/YOLOv12 in OpenVINO IR format outputs a single tensor with shape:

```
(1, 4 + num_classes, N)
```

Where:
- `1` = batch size
- `4` = [cx, cy, w, h] bounding box coordinates in letterboxed pixel space
- `num_classes` = 1 for 2G model, 3 for 3G/4G model
- `N` = number of anchor grid points (~8400 for 640x640 input)

The first element `output[0]` extracts the batch dimension, giving shape `(4+nc, N)`.

### 6.5 Postprocessing

```python
def postprocess_yolo_output(output, conf_threshold, letterbox_info, iou_threshold=0.45):
    raw = output[0]            # (4+nc, N)
    predictions = raw.T        # (N, 4+nc) - transpose for row-wise iteration

    # Filter by confidence
    class_scores = predictions[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    mask = max_scores > conf_threshold
    predictions = predictions[mask]
    max_scores = max_scores[mask]

    if len(predictions) == 0:
        return []

    class_ids = np.argmax(predictions[:, 4:], axis=1)
    boxes = predictions[:, :4].copy()

    # Remove letterbox transform: undo padding and scale
    scale = letterbox_info['scale']
    pad_w = letterbox_info['pad_w']
    pad_h = letterbox_info['pad_h']
    boxes[:, 0] = (boxes[:, 0] - pad_w) / scale  # cx
    boxes[:, 1] = (boxes[:, 1] - pad_h) / scale  # cy
    boxes[:, 2] = boxes[:, 2] / scale             # w
    boxes[:, 3] = boxes[:, 3] / scale             # h

    # Convert center-format to corner-format for NMS
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # Per-class NMS
    for cls_id in range(num_classes):
        cls_indices = np.where(class_ids == cls_id)[0]
        cls_keep = numpy_nms(x1[cls_indices], y1[cls_indices],
                             x2[cls_indices], y2[cls_indices],
                             max_scores[cls_indices], iou_threshold)
        ...
```

**Confidence thresholds by model**:
- 3G/4G model: `conf_threshold=0.6` - higher threshold because 3G/4G signals have distinctive wide spectral shapes with strong evidence
- 2G model: `conf_threshold=0.3` - lower threshold because 2G GSM signals are narrow (200 kHz) and can produce lower-confidence activations

**IOU threshold**: `iou_threshold=0.45` for both models. Boxes with intersection-over-union above this value are suppressed, keeping only the highest-confidence detection per overlapping group.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`, lines 155-217

### 6.6 Non-Maximum Suppression

```python
def numpy_nms(x1, y1, x2, y2, scores, iou_threshold):
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]   # Sort by confidence descending
    keep = []
    while len(order) > 0:
        i = order[0]                  # Always keep the best box
        keep.append(i)
        if len(order) == 1:
            break
        # Compute IoU of best box against all remaining
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]
    return keep
```

NMS is performed **per class** in `postprocess_yolo_output`. This means 3G boxes only suppress other 3G boxes, and 4G boxes only suppress other 4G boxes. A 3G and 4G signal at the same frequency location will both be reported. This is correct behavior because a site can genuinely transmit both 3G and 4G on adjacent channels.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`, lines 132-152

### 6.7 Model Warmup

After loading, both models are warmed up with a dummy inference:

```python
# 2G warmup with small blob (real 2G images have few rows)
dummy_blob = np.zeros((1, 3, 64, 640), dtype=np.float32)
model_2g_compiled([dummy_blob])[model_2g_output]

# 3G/4G warmup at actual input size
dummy_blob = np.zeros((1, 3, model_3g_4g_h, model_3g_4g_w), dtype=np.float32)
model_3g_4g_compiled([dummy_blob])[model_3g_4g_output]
```

**Rationale**: OpenVINO compiles optimized execution kernels on first inference. Without warmup, the first real request incurs JIT compilation overhead that can add seconds of latency. The warmup runs before `s.listen(1)` so the TCP server only becomes available after kernels are compiled.

The 2G warmup uses a 64-row height rather than 640 because the actual 2G input images are much shorter than the model's maximum supported height. Using a small warmup blob ensures that the small-height kernel path (which may be code-generated differently) is also warmed.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`, lines 259-268

---

## 7. AI Models

### 7.1 Model Overview

| Attribute | 2G Model | 3G/4G Model |
|-----------|----------|-------------|
| Architecture | YOLO11n | YOLOv12n |
| Source | `best.pt` (50 MB) | `best.pt` (5.3 MB) |
| OpenVINO FP32 | `best_openvino_model/` (99 MB) | `best_openvino_model/` (21 MB) |
| Classes | 1: `2G` | 3: `3G`, `4G`, `4G-TDD` |
| Input size (metadata) | 640x640 | 640x640 |
| Export config | `dynamic=True, half=False, int8=False` | same |
| Stride | 32 | 32 |
| Confidence threshold | 0.3 | 0.6 |
| Training date | 2026-02-20 | 2026-02-20 |

### 7.2 Class Definitions

**2G Model (YOLO11n)**:
- Class 0: `2G` - GSM (Global System for Mobile), 200 kHz channel spacing

**3G/4G Model (YOLOv12n)**:
- Class 0: `3G` - UMTS/WCDMA, 5 MHz channel bandwidth
- Class 1: `4G` - LTE FDD (Frequency Division Duplex), 5–20 MHz bandwidth
- Class 2: `4G-TDD` - LTE TDD (Time Division Duplex), predominantly used in Band 40 (2300–2400 MHz)

The `add_detected_cells_to_list` function maps class IDs to output lists:
- `cls==0` from 3G/4G model -> `predicted_freq_list_3g`
- `cls==1 or cls==2` from 3G/4G model -> `predicted_freq_list_4g`
- Any detection from 2G model -> `predicted_freq_list_2g`

### 7.3 Model Selection and Fallback

At startup, the 2G model is selected using a priority chain:

```python
if os.path.isdir("2G_MODEL/best_openvino_model"):
    # FP32 preferred: higher accuracy for weak 2G signals
    model_2g_compiled, model_2g_output, model_2g_shape = load_openvino_model(
        "2G_MODEL/best_openvino_model/")
elif os.path.isdir("2G_MODEL/best_int8_openvino_model"):
    # INT8 fallback: acceptable for stronger signals
    model_2g_compiled, model_2g_output, model_2g_shape = load_openvino_model(
        "2G_MODEL/best_int8_openvino_model/")
else:
    raise FileNotFoundError("No 2G model found in 2G_MODEL/")
```

**Why FP32 over INT8**: The 2G model detects narrow-band (200 kHz) signals that may appear at low power relative to the noise floor or adjacent 3G/4G transmitters. INT8 quantization reduces each weight from 32 bits to 8 bits by mapping float values to integer ranges. This quantization introduces rounding errors that disproportionately affect the fine spatial features the 2G model relies on. In testing, INT8 quantization caused weak-signal 2G detections to be missed entirely while strong-signal detections remained accurate.

The 3G/4G model has no INT8 variant because 3G/4G signals are substantially wider (5–20 MHz) and produce robust high-confidence activations that are tolerant of quantization noise.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`, lines 232-242

### 7.4 Input Shape Determination

The models were exported with `dynamic=True`, which means their OpenVINO IR input tensors have dynamic dimensions (expressed as `?` in the partial shape). The actual intended input size is stored in `metadata.yaml`:

```yaml
# 2G_MODEL/best_openvino_model/metadata.yaml
imgsz:
- 640
- 640

# 3G_4G_MODEL/best_openvino_model/metadata.yaml
imgsz:
- 640
- 640
```

However, because the 2G model receives variable-height images (the composite gap image height matches the spectrogram height, which varies by request), the `auto=True` letterbox mode produces stride-aligned input sizes that differ from 640x640 for each request. The 640x640 from metadata is used only as the maximum target dimension for scale calculation.

---

## 8. TCP Server and Protobuf Protocol

### 8.1 Server Startup

```python
def main():
    scanner_ai_host = os.getenv('SCANNER_AI_IP', '0.0.0.0')
    scanner_ai_port = int(os.getenv('SCANNER_AI_PORT', 4444))
    s.bind((scanner_ai_host, scanner_ai_port))
    s.listen(1)
    while True:
        connection, _ = s.accept()
        reciever_thread = threading.Thread(
            target=recieve_samples,
            args=(connection, 1024, scanner_ai_save_samples, memory_optimization))
        reciever_thread.start()
```

`SO_REUSEADDR` is set on the socket to allow immediate restart after process termination without waiting for TIME_WAIT expiration:

```python
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
```

Each accepted connection is handled in a new daemon thread, allowing concurrent connections (though each inference is CPU-bound and they will serialize on CPU resources).

### 8.2 Protobuf Message Protocol

The protocol uses two-message exchanges:

**Exchange 1: Metadata + Acknowledgment**

```
Client -> Server: AIModelReq { predict_sample_req {
    id: <band_id>,
    sampling_rate_khz: 30720,
    center_freq_khz: <center>,
    bw_khz: <bandwidth>,
    num_chunks: <num_center_frequencies>,
    overlay_khz: <overlap>,
    samples_len: <byte_length_of_next_message>
} }

Server -> Client: AIModelRes { predict_sample_res {
    result: AI_RESULT_SUCCESS_UNSPECIFIED,
    id: <band_id>
} }
```

The `samples_len` field tells the server how many bytes to expect in the next message, enabling exact-length socket reads.

**Exchange 2: Sample Data + Predictions**

```
Client -> Server: AIModelReq { sample_data_req {
    id: <band_id>,
    samples: [<float32 values...>]
} }

Server -> Client: AIModelRes { sample_data_res {
    id: <band_id>,
    lte_freqs: [<4G MHz values...>],
    umts_freqs: [<3G MHz values...>],
    gsm_freqs: [<2G MHz values...>]
} }
```

### 8.3 Socket Read Loop

The sample data may be larger than a single `recv()` call can deliver (TCP is a stream protocol). The server reads in chunks until all `sample_len` bytes are received:

```python
bytes_recd = 0
chunks = []
while bytes_recd < sample_len:
    chunk = conn.recv(min(sample_len - bytes_recd, 65000))
    chunks.append(chunk)
    bytes_recd += len(chunk)
scanner_ai_data_req_1.ParseFromString(b''.join(chunks))
```

The 65000-byte chunk size is slightly below the maximum UDP datagram size and is a common choice for TCP reads as it fits comfortably in most socket buffer implementations.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`, lines 525-608

### 8.4 SIGTERM Handling

```python
def sigterm_handler(_signo, _stack_frame):
    logging.info("Received SIGTERM, exiting...")
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)
```

Docker sends SIGTERM before SIGKILL on container stop. This handler ensures a clean log message and immediate exit rather than waiting for the default 10-second SIGTERM timeout.

---

## 9. Colormap and Power Normalization

### 9.1 Module Structure

```
ai_colormap.py
|-- NormalizePowerValue    # dBm -> integer index [0, 254]
`-- CustomImg              # integer index -> RGB pixel via viridis lookup

viridis_colormap.py
`-- var: np.ndarray        # shape (256, 3), dtype int, values [0, 255]
                           # 256 pre-computed RGB color triplets
```

### 9.2 Why Pre-computed Colormap

The standard approach would be to use `matplotlib.cm.viridis(value)`. However, importing matplotlib at runtime adds approximately 200 MB of dependencies and significant import time. Instead, `viridis_colormap.py` contains the 256 pre-computed RGB triplets that the viridis colormap produces at evenly spaced intervals from 0 to 1.

The lookup operation becomes a single NumPy fancy indexing call:

```python
return self.colors[img.astype(int)]   # img shape: [H, W], output: [H, W, 3]
```

This is extremely efficient because NumPy implements fancy indexing as a vectorized memory gather operation.

### 9.3 Normalization Range Validation

The normalization clips to `[-130, -3]` dBm. Values outside this range are edge cases:
- Below -130 dBm: Below the hardware noise floor; clipped to noise floor color
- Above -3 dBm: Rare (implies extremely strong signal or hardware saturation); clipped to maximum-signal color

The 0.5 dBm step size produces `(-3 - (-130)) / 0.5 = 254` distinct index values, filling the [0, 254] range.

**File reference**: `/Users/manip/Documents/codeRepo/scanner_ai_v2/ai_colormap.py`

---

## 10. Migration: Ultralytics to OpenVINO

### 10.1 Overview

This section documents the architectural changes made to migrate the scanner AI system from the Ultralytics YOLO runtime to Intel's OpenVINO inference runtime. The migration was driven by deployment size constraints, dependency management complexity, and the need for a runtime that does not require PyTorch.

### 10.2 Before: Ultralytics Runtime

The original implementation used Ultralytics as the runtime inference library:

```python
# Original approach (pre-migration)
from ultralytics import YOLO

model_2g = YOLO("2G_MODEL/best.pt")
model_3g_4g = YOLO("3G_4G_MODEL/best.pt")

# Inference was a single call; Ultralytics handled everything internally
results = model_2g.predict(image, conf=0.3, iou=0.45)

# Results accessed via Ultralytics result objects
for box in results[0].boxes:
    cls = int(box.cls)
    cx, cy, w, h = box.xywh[0].tolist()
    conf = float(box.conf)
```

**What Ultralytics handled automatically:**
1. PyTorch model loading from `.pt` (checkpoint) files
2. GPU/CPU device selection
3. Input preprocessing (letterbox resize, normalization, CHW conversion)
4. Tensor inference
5. Output postprocessing (confidence filtering)
6. Non-Maximum Suppression (NMS)
7. Result object wrapping

**Runtime dependency chain:**
```
ultralytics 8.x
  -> torch >= 1.8         (~800 MB)
  -> torchvision           (~200 MB)
  -> numpy, PIL, opencv    (~100 MB)
  -> pyyaml, psutil, etc.
Total: ~1.1–1.3 GB installed
```

### 10.3 After: OpenVINO Runtime

The migrated implementation uses OpenVINO exclusively:

```python
# New approach (post-migration)
import openvino as ov

_ov_core = ov.Core()
model = _ov_core.read_model("2G_MODEL/best_openvino_model/best.xml")
compiled_model = _ov_core.compile_model(model, "CPU")

# All preprocessing/postprocessing is explicit custom code
blob, letterbox_info = preprocess_image(img, target_h, target_w, auto=True)
raw_output = compiled_model([blob])[output_layer]
detections = postprocess_yolo_output(raw_output, conf_threshold, letterbox_info)
```

**Runtime dependency chain:**
```
openvino >= 2024.0.0       (~180 MB)
opencv-python-headless      (~35 MB)
numpy                       (~25 MB)
protobuf                    (~5 MB)
Pillow                      (~5 MB)
Total: ~250 MB installed
```

**Savings: ~850 MB to 1 GB reduction in installation size.**

### 10.4 What Had to Be Re-implemented

When Ultralytics was removed, all functionality it provided internally had to be re-implemented explicitly. The following custom functions replaced Ultralytics internals:

#### 10.4.1 Letterbox Preprocessing (`preprocess_image`)

Ultralytics letterbox transform is reproduced manually. The key detail is the `auto=True` mode which produces stride-aligned padding - this matches exactly what Ultralytics uses internally for dynamic-shape models.

The `114.0/255.0` gray fill value matches Ultralytics' default letterbox fill color.

#### 10.4.2 Output Tensor Parsing (`postprocess_yolo_output`)

In Ultralytics, the YOLO head's NMS-excluded output tensor has shape `(batch, 4+nc, anchors)`. The transposition `raw.T` converts this to `(anchors, 4+nc)` for row-wise processing. This shape convention is specific to YOLO11/YOLOv12 exported without end-to-end NMS (`end2end: false` in metadata).

The coordinate system in the raw tensor is letterboxed pixel space (cx, cy, w, h relative to the padded canvas). The `postprocess_yolo_output` function undoes the letterbox transform to return coordinates in original image pixel space.

#### 10.4.3 Non-Maximum Suppression (`numpy_nms`)

Ultralytics uses a C++/CUDA-accelerated NMS kernel from torchvision. The replacement `numpy_nms` implements the standard greedy NMS algorithm purely in NumPy. For the small number of candidate detections typical in this domain (< 50 per inference), pure-NumPy NMS performance is equivalent.

The per-class NMS loop in `postprocess_yolo_output` matches Ultralytics' default behavior of class-aware NMS.

### 10.5 Behavioral Differences Introduced by Migration

| Behavior | Ultralytics | OpenVINO |
|----------|------------|----------|
| Preprocessing | Implicit, tested by library | Explicit custom code, must match training |
| NMS | C++ accelerated | NumPy, functionally identical |
| Coordinate system | Original image space (auto-mapped) | Letterboxed space (must undo manually) |
| Confidence filtering | Integrated with NMS | Applied before NMS as separate step |
| Class handling | Multi-label supported | Single argmax class per box |
| Model format | `.pt` (PyTorch checkpoint) | `.xml` + `.bin` (OpenVINO IR) |
| Dynamic shapes | Automatic | Requires metadata.yaml parsing |
| Device selection | Auto (CUDA if available) | Explicit CPU device |

### 10.6 Issues Resolved During Migration

The migration commit message (`3281651`) documents several bugs fixed alongside the runtime migration:

**Bug 1: Thread Safety - Global Mutable State**

Before: Detection result lists (`predicted_freq_list_4g`, etc.) were module-level global variables. Multiple concurrent TCP connections would corrupt each other's results.

After: Lists are now local variables within `predict_samples()`, which is called within each connection's thread. Each request gets its own independent result lists.

```python
# Before (global state - bug)
predicted_freq_list_4g = []  # module level

# After (thread-local - fixed)
def predict_samples(...):
    predicted_freq_list_4g = []  # local to function
    predicted_freq_list_3g = []
    predicted_freq_list_2g = []
```

**Bug 2: Socket Port Reuse**

Before: Restarting the service within 60 seconds of stopping it caused "Address already in use" because the OS holds TCP connections in TIME_WAIT state.

After: `SO_REUSEADDR` is set before `bind()`, allowing immediate reuse of the port.

**Bug 3: Empty Connection Handling**

Before: A TCP connection that sends no data would cause an unhandled exception.

After: An empty `recv()` result is detected and the connection is closed gracefully.

**Bug 4: `np.append` Performance**

Before: `colormapped_array_2G = np.append(...)` was used in the 2G image assembly loop. `np.append` creates a new full copy of the array on every call, giving O(n²) memory allocation behavior.

After: `np.concatenate((...), axis=1)` is used. This is semantically identical but takes a tuple of arrays and concatenates in one pass, which is the documented correct usage for building arrays incrementally.

**Bug 5: Pre-allocated Singletons**

Before: `NormalizePowerValue()` and `CustomImg()` were instantiated on every call to `predict_samples`.

After: Single instances are created at module load time:
```python
_normalizer = ai_colormap.NormalizePowerValue()
_colormap = ai_colormap.CustomImg()
```

### 10.7 Test Expectation Updates

The integration tests (`testing/test_scanner_ai_script.py`) were updated because OpenVINO inference results differ slightly from Ultralytics PyTorch results. The differences are within 0.1–0.5 MHz and are caused by:

1. FP32 floating-point operation ordering differences between PyTorch and OpenVINO CPU backends
2. The explicit custom preprocessing matching the Ultralytics internal implementation but not being bit-for-bit identical

A frequency tolerance of ±1.0 MHz was introduced in `freq_is_detected()`:

```python
FREQ_TOLERANCE_MHZ = 1.0

def freq_is_detected(expected_freq, detected_list, tolerance=FREQ_TOLERANCE_MHZ):
    return any(abs(expected_freq - det) <= tolerance for det in detected_list)
```

This tolerance is appropriate for the use case: the scanner system uses detected frequencies as starting candidates for further decoding, not as final measurement values.

---

## 11. Migration Advantages and Known Issues

This section provides a consolidated view of the benefits gained and the trade-offs accepted by migrating from Ultralytics/PyTorch to OpenVINO.

### 11.1 Advantages

#### A1. Dramatic Reduction in Runtime Dependencies

| Metric | Ultralytics | OpenVINO | Improvement |
|--------|------------|----------|-------------|
| Installed size | ~1.1–1.3 GB | ~250 MB | **~80% smaller** |
| Python packages | ~45 transitive deps | ~8 packages | **~80% fewer** |
| Docker image size | ~2.5 GB | ~800 MB | **~68% smaller** |
| `pip install` time | ~3–5 min | ~30–45 sec | **~5x faster** |

Removing PyTorch alone eliminates ~800 MB. This directly reduces container pull times, disk usage on embedded devices, and CI build durations.

#### A2. No GPU/CUDA Dependency

Ultralytics pulls `torch` which ships CUDA stubs and tries to detect GPU hardware at import time. On CPU-only servers (which is the production deployment for this scanner), this adds unused bloat and occasional import warnings. OpenVINO targets CPU natively with no GPU detection overhead.

#### A3. Faster Cold Start

| Phase | Ultralytics | OpenVINO |
|-------|------------|----------|
| Python import | ~4–6 sec (torch + ultralytics) | ~1–2 sec (openvino) |
| Model loading | ~2–3 sec (.pt deserialization + JIT) | ~1–2 sec (.xml/.bin direct load) |
| First inference | ~1 sec (graph compilation) | ~0.5 sec (pre-compiled) |
| **Total startup** | **~7–11 sec** | **~3–5 sec** |

The OpenVINO IR format (`.xml` + `.bin`) is already an optimized graph representation. PyTorch `.pt` files contain raw checkpoint state that requires model reconstruction and optimization at load time.

#### A4. Optimized CPU Inference

OpenVINO is specifically designed for CPU inference optimization:

- **Intel MKL-DNN kernels**: Optimized matrix multiplication and convolution for x86 CPUs
- **Graph fusion**: Combines consecutive operations (Conv + BatchNorm + ReLU → single fused op)
- **Memory layout optimization**: Uses optimal tensor memory layout (blocked format) for CPU cache hierarchy
- **Thread pool**: Dedicated inference thread pool with LATENCY hint avoids thread contention

In practice, inference latency is comparable or slightly better than PyTorch CPU for this model size range (5–100 MB).

#### A5. Deterministic and Reproducible Builds

With Ultralytics, `pip install ultralytics` could pull different PyTorch versions across builds, leading to subtle inference differences. The OpenVINO approach uses:

- Pre-exported `.xml/.bin` files (frozen model graph)
- Pinned `openvino>=2024.0.0` version
- `uv.lock` for exact dependency resolution
- No model conversion or JIT compilation at runtime

#### A6. Simpler Error Surface

Ultralytics wraps hundreds of configuration options, automatic behaviors, and silent fallbacks. Debugging inference issues required understanding Ultralytics internals. With OpenVINO:

- Preprocessing is explicit and visible in `preprocess_image()`
- Postprocessing is explicit in `postprocess_yolo_output()`
- NMS is a standalone function `numpy_nms()` that can be unit-tested
- No hidden configuration files or environment variables from the library

#### A7. No License Ambiguity

Ultralytics uses AGPL-3.0 for its open-source version, which requires derivative works to be open-sourced. OpenVINO uses Apache-2.0, which is permissive and compatible with proprietary deployment.

#### A8. Thread-Safe by Design

The shared `ov.Core()` instance is documented as thread-safe by Intel. Multiple TCP connections can run concurrent inferences without locking the model. Ultralytics' `YOLO.predict()` was not explicitly documented for thread safety and required careful handling.

---

### 11.2 Known Issues and Trade-offs

#### I1. Custom Preprocessing Must Exactly Match Training

**Severity: High** | **Status: Resolved**

The most critical risk in the migration. YOLO models are sensitive to preprocessing - if the letterbox padding, normalization, or color channel order differs from what was used during training, detection accuracy degrades silently (no errors, just missed detections).

**What can go wrong:**
- Wrong fill color in letterbox (must be `114/255 ≈ 0.447`, not `0.0` or `0.5`)
- Wrong channel order (model expects RGB but receives BGR, or vice versa)
- Wrong normalization range (must be `[0.0, 1.0]`, not `[-1.0, 1.0]`)
- Wrong padding strategy (`auto=True` for stride-aligned vs `auto=False` for fixed-size)

**Mitigation:** The `preprocess_image()` function was carefully matched against Ultralytics source code (`ultralytics/data/augment.py:LetterBox`). Integration tests across 6 frequency bands validate that detection results match the expected frequencies within ±1.0 MHz tolerance.

#### I2. NMS Implementation Differences

**Severity: Low** | **Status: Accepted**

The custom `numpy_nms()` uses the standard greedy IoU-based NMS algorithm. Ultralytics uses torchvision's C++ NMS which may handle edge cases (identical scores, exact overlap) differently due to floating-point ordering.

**Impact:** In practice, no detection differences have been observed because:
- The scanner domain produces few overlapping detections (< 50 per image)
- Confidence scores are well-separated (strong signals >> noise floor)
- The ±1.0 MHz tolerance absorbs any marginal NMS ordering differences

#### I3. No Automatic Model Format Updates

**Severity: Medium** | **Status: Accepted trade-off**

With Ultralytics, upgrading the library version could transparently support new model architectures (YOLOv9, v10, v11, v12) from `.pt` files. With OpenVINO:

- New model architectures require re-running `export_openvino.py` to produce updated `.xml/.bin`
- If a new YOLO version changes the output tensor format, `postprocess_yolo_output()` must be updated manually
- The `export_openvino.py` script still requires `ultralytics` to be installed (but only on the export machine, not production)

**Mitigation:** Model exports are versioned alongside code. The `metadata.yaml` embedded in each export contains the Ultralytics version and model architecture details.

#### I4. INT8 Quantization Loses Weak 2G Detections

**Severity: Medium** | **Status: Mitigated (FP32 default)**

INT8 quantization (8-bit integer weights) reduces the 2G model from 103 MB to 26 MB and speeds up inference. However, weak 2G/GSM signals near the noise floor are missed by the INT8 model because quantization reduces the dynamic range of feature activations.

**Root cause:** Generic calibration data (e.g., COCO images) was used for quantization instead of domain-specific spectrogram data. The power distribution of spectrograms is very different from natural images.

**Mitigation:** FP32 is the default. The code prioritizes `best_openvino_model/` (FP32) over `best_int8_openvino_model/` (INT8). INT8 is only used as a fallback if FP32 is not available.

**Future fix:** Re-quantize with spectrogram-specific calibration data using NNCF (Neural Network Compression Framework).

#### I5. CPU-Only Inference

**Severity: Low** | **Status: Accepted by design**

OpenVINO as configured only supports CPU inference. If GPU acceleration is ever needed:

- OpenVINO supports Intel integrated GPUs via the GPU plugin
- NVIDIA GPUs would require switching to TensorRT or ONNX Runtime with CUDA
- The current production hardware is CPU-only, so this is not a limitation today

#### I6. No Built-in Model Validation

**Severity: Low** | **Status: Mitigated by tests**

Ultralytics provides `model.val()` to evaluate mAP/precision/recall on a validation dataset. With OpenVINO, there is no built-in validation tool. Model accuracy must be validated through:

- The 6-band integration test suite (covers known frequency detections)
- Manual inspection of spectrogram images (saved when `SAVE_SAMPLES=YES`)
- Comparison against Ultralytics results during export (one-time)

#### I7. Slight Frequency Accuracy Drift

**Severity: Low** | **Status: Accepted**

OpenVINO inference results differ from Ultralytics/PyTorch by 0.1–0.5 MHz due to floating-point operation ordering differences between the two runtimes. This is within the acceptable ±1.0 MHz tolerance for the downstream frequency decoding stage.

**Cause:** FP32 arithmetic is not associative. Different backends may compute `(a + b) + c` vs `a + (b + c)`, producing results that differ at the least-significant bits. These small differences accumulate through the 50+ convolution layers in the YOLO network.

#### I8. Metadata Parsing Fragility

**Severity: Low** | **Status: Accepted**

The `_read_model_imgsz()` function parses `metadata.yaml` with a simple line-by-line parser instead of a proper YAML library. This avoids adding `pyyaml` as a dependency but means:

- Non-standard YAML formatting could break parsing
- Only the `imgsz` field is extracted; other metadata is ignored
- If parsing fails, the model falls back to 640x640 (which may not match the actual training size)

**Mitigation:** The export script (`export_openvino.py`) always produces standard Ultralytics metadata format. The fallback to 640x640 is safe because both models were trained at 640x640.

---

### 11.3 Summary: Risk Assessment Matrix

| Issue | Severity | Likelihood | Impact | Status |
|-------|----------|-----------|--------|--------|
| I1: Preprocessing mismatch | High | Low (tested) | Detection failures | Resolved |
| I2: NMS differences | Low | Very low | ±0.1 MHz drift | Accepted |
| I3: Manual model updates | Medium | On retrain | Dev effort | Accepted |
| I4: INT8 weak detection loss | Medium | Only if INT8 used | Missed 2G signals | Mitigated (FP32 default) |
| I5: CPU-only | Low | N/A (by design) | No GPU acceleration | Accepted |
| I6: No built-in validation | Low | On retrain | Needs manual testing | Mitigated by tests |
| I7: Frequency drift | Low | Every inference | ±0.5 MHz shift | Accepted |
| I8: YAML parser fragility | Low | Very low | Fallback to 640x640 | Accepted |

### 11.4 Recommendation

The migration is a net positive for this project. The advantages (A1–A8) significantly outweigh the issues (I1–I8). The only high-severity issue (preprocessing mismatch) has been resolved and is protected by integration tests. All remaining issues are either by-design trade-offs or have acceptable mitigations in place.

For future work, the main area to watch is **I3 (manual model updates)** - when new YOLO architectures are adopted, the export and postprocessing code must be updated in sync.

---

## 12. Model Export Process

### 11.1 Export Script

The `export_openvino.py` script converts trained PyTorch `.pt` files to OpenVINO IR format. This script is not part of the runtime deployment and is only needed when retraining produces new `.pt` files.

```python
from ultralytics import YOLO

MODEL_DIRS = ["2G_MODEL", "3G_4G_MODEL"]
IMGSZ = 640

def export_model(pt_path):
    model = YOLO(pt_path)
    export_path = model.export(
        format="openvino",
        imgsz=IMGSZ,
        half=False,       # FP32 (not FP16) for maximum accuracy
        int8=False,       # No INT8 quantization
        dynamic=True,     # Variable input dimensions
    )
```

### 11.2 Export Configuration Decisions

**`format="openvino"`**: Exports directly to OpenVINO IR format. Ultralytics internally converts `.pt` -> ONNX -> OpenVINO IR using OpenVINO's model optimizer (`mo`).

**`imgsz=640`**: Sets the model's reference input size for static shape models and the maximum dimension hint for dynamic shape models. 640 is the standard YOLO training size.

**`half=False`**: Disables FP16 export. FP16 (half precision) would halve the model file size but introduces precision loss. Given the importance of detecting weak 2G signals, FP32 is retained.

**`int8=False`**: Disables INT8 post-training quantization. If INT8 were needed for deployment size, a calibration dataset of representative spectrogram samples would be required to minimize accuracy loss.

**`dynamic=True`**: Exports with dynamic input shape dimensions. This is essential for the 2G model because the 2G inference image's height and width are determined by the spectrogram dimensions at runtime, which vary by request. Without `dynamic=True`, the model would require fixed 640x640 inputs and any other size would require an additional resize.

### 11.3 Output Structure

After export, each model directory contains:

```
best_openvino_model/
|-- best.xml           # Model topology (graph definition in XML)
|-- best.bin           # Model weights (raw binary tensor data)
|-- metadata.yaml      # Ultralytics metadata: imgsz, class names, export args
`-- model_metadata.json  # Custom provenance record (added during export)
```

The `.xml` and `.bin` are the two required files for OpenVINO. The `.bin` file is referenced by path inside `.xml` and must remain alongside it.

### 11.4 Why Export Separate from Runtime

The export step requires Ultralytics (and therefore PyTorch) to be installed. By separating export from inference:

1. The production Docker image does not need PyTorch at all
2. Export can be run on a developer machine with GPU acceleration for faster model training/evaluation
3. The OpenVINO IR files are committed to the repository as build artifacts, making deployments reproducible without requiring the original PyTorch environment

This is a common pattern called "build-time vs. runtime dependencies" applied to ML model deployment.

---

## 13. Docker and Deployment

### 12.1 Multi-Stage Dockerfile

```dockerfile
## Builder Stage
FROM ubuntu:22.04 as builder

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential curl ca-certificates

# Install uv package manager
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod -R 655 /install.sh && /install.sh && rm /install.sh

ENV PATH="/root/.local/bin:${PATH}"
ENV UV_PYTHON_INSTALL_DIR=/app/python

WORKDIR /app
COPY ./.python-version .
COPY ./pyproject.toml .
RUN uv sync   # Resolves and installs all dependencies

## Production Stage
FROM ubuntu:22.04 AS production

RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 libglib2.0-0   # OpenCV runtime requirements

WORKDIR /app

# Copy only compiled venv and Python runtime from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/python /app/python

# Copy only OpenVINO IR models (NOT .pt files)
COPY 2G_MODEL/best_openvino_model /app/2G_MODEL/best_openvino_model
COPY 3G_4G_MODEL/best_openvino_model /app/3G_4G_MODEL/best_openvino_model
COPY ./*.py /app

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 4444
STOPSIGNAL SIGINT
CMD ["python3", "scanner.py"]
```

**Key design decisions:**

**Multi-stage build**: The builder stage installs `build-essential` and `curl` for package compilation. The production stage only has `libgl1` and `libglib2.0-0` (OpenCV's shared library dependencies). This keeps the production image lean.

**`uv` package manager**: `uv` is significantly faster than `pip` for dependency resolution and installation. The `uv.lock` file provides reproducible installs. The `UV_PYTHON_INSTALL_DIR=/app/python` makes `uv` install its managed Python runtime to a path that can be copied between stages.

**No `.pt` files**: Only `best_openvino_model/` directories are copied, not the `.pt` source weights. This saves 55 MB from the image (50 MB for 2G `.pt` + 5.3 MB for 3G/4G `.pt`).

**`STOPSIGNAL SIGINT`**: Docker uses SIGINT for graceful shutdown. The `sigterm_handler` in `scanner.py` handles SIGTERM as well, covering both Docker's SIGTERM (on `docker stop`) and Kubernetes pod termination.

### 12.2 docker-compose Service Definition

```yaml
services:
  cellularscanner:
    image: "docker.artifactory.internal/cellularscanner/cellularscanner_ai:${SCANNER_AI_VERSION}"
    container_name: cellular_scanner_ai
    ports:
      - 4444:4444
    volumes:
      - ${SCANNER_AI_LOW_POWER_SAMPLES}:/app/SAMPLES_LOW_POWER
    environment:
      - SCANNER_AI_PORT=4444
      - SAVE_SAMPLES=NO
      - MEM_OPTIMIZATION=YES
    ulimits:
      core: -1       # Allow core dumps for debugging crashes
    restart: always
    networks:
      - docker-network
```

The `core: -1` ulimit allows unlimited core dump file size. This enables post-mortem debugging if the Python process crashes due to a segfault in the OpenVINO C++ layer.

The `SCANNER_AI_VERSION` environment variable controls which image tag is deployed, matching the CI/CD pattern of the broader OneShell POS monorepo.

### 12.3 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCANNER_AI_IP` | `0.0.0.0` | Bind address for TCP server |
| `SCANNER_AI_PORT` | `4444` | TCP server port |
| `SAVE_SAMPLES` | (unset) | `"YES"` to save spectrogram images to `SAMPLES_LOW_POWER/` |
| `MEM_OPTIMIZATION` | (unset) | `"YES"` to enable 60 MHz chunking for bands >= 100 MHz |

---

## 14. Configuration and Environment Variables

### 13.1 Runtime Configuration

All configuration is provided via environment variables. There is no configuration file for the runtime; the `settings.json` file in the repository is a development-only VS Code settings file.

### 13.2 FFT Constants

These constants are defined at module level in `scanner.py` and match the hardware's FFT configuration:

```python
fft_size = 2048          # FFT points per sweep
num_khz_per_fft_point = 15   # 15 kHz per FFT bin (30.72 MHz / 2048 = 15 kHz)
fifteen_mhz_points = int(15000 / 15) = 1000   # bins for 15 MHz
five_mhz_points = int(5000 / 15) = 333        # bins for 5 MHz
```

These are derived from the scanner hardware's 30.72 MHz sample rate, which is standard for LTE (it is exactly 2 × 15.36 MHz, the LTE chip rate).

### 13.3 Memory Optimization Parameters

```python
MAX_AI_BANDWIDTH = 60 * 1e3   # 60 MHz per inference chunk (in kHz)
overlap = 12 MHz               # Chunk overlap to prevent edge effects
```

The 60 MHz inference window and 12 MHz overlap were determined empirically to match the frequency range represented in the training dataset images. Changing these values would require model retraining.

---

## 15. Testing

### 14.1 Test Structure

```
testing/test_scanner_ai_script.py
|-- Bands_to_be_tested: [1, 3, 8, 20, 28, 40]
|-- expected_4g_frequencies: {band -> [MHz, ...]}
|-- expected_3g_frequencies: {band -> [MHz, ...]}
|-- expected_2g_frequencies: {band -> [MHz, ...]}
|-- bandwise_parameters: {band -> [center_khz, bw_khz, num_chunks]}
|-- freq_is_detected()    # Tolerance-based match check
|-- samples_test()        # Full TCP round-trip for one band
`-- test_check_detected_freq()  # pytest entry point
```

### 14.2 Test Execution Requirements

Tests require a running Scanner AI server on `127.0.0.1:4444`. The test suite:
1. Connects via TCP socket to the running server
2. Sends real spectrogram binary data from `SAMPLES_UT/sample_vec_B{band}.dat`
3. Receives detection results
4. Asserts all expected frequencies are present within ±1.0 MHz tolerance

Tests must be run with the server already running:
```bash
# Terminal 1
python3 scanner.py

# Terminal 2
python3 -m pytest testing/test_scanner_ai_script.py -v
```

### 14.3 Test Coverage by Band

| Band | Frequency Range | Expected 4G | Expected 3G | Expected 2G |
|------|----------------|-------------|-------------|-------------|
| 1 | 2110–2170 MHz (DL) | 2165.0, 2146.7 | 2116.4, 2137.7 | None |
| 3 | 1805–1880 MHz (DL) | 1815.0, 1870.0, 1849.5 | None | 1860.2 |
| 8 | 925–960 MHz (DL) | None | 932.6, 937.2, 927.5 | 953.4 |
| 20 | 791–821 MHz (DL) | 813.6, 798.5 | None | None |
| 28 | 758–803 MHz (DL) | 763.1, 800.8 | None | None |
| 40 | 2300–2400 MHz (TDD) | 2342.1, 2361.9 | None | None |

Band 40 exercises the memory optimization chunking path because its 100 MHz bandwidth equals the chunking threshold.

### 14.4 Band Parameters

```python
bandwise_parameters = {
    "Band_1":  [2140000, 60000,  5],  # center=2140 MHz, bw=60 MHz, 5 sweeps
    "Band_3":  [1845000, 80000,  7],  # center=1845 MHz, bw=80 MHz, 7 sweeps
    "Band_8":  [945000,  40000,  3],  # center=945 MHz,  bw=40 MHz, 3 sweeps
    "Band_20": [806000,  30000,  2],  # center=806 MHz,  bw=30 MHz, 2 sweeps
    "Band_28": [783000,  50000,  4],  # center=783 MHz,  bw=50 MHz, 4 sweeps
    "Band_40": [2350000, 100000, 9],  # center=2350 MHz, bw=100 MHz, 9 sweeps
}
```

All values are in kHz. `overlay_khz=10000` (10 MHz overlap) is used for all test bands.

---

## 16. Performance Characteristics

### 15.1 Observed Latency (Apple M2)

| Band | Bandwidth | Chunks | Inference Time |
|------|-----------|--------|---------------|
| Band 20 | 30 MHz | 1 | ~237 ms |
| Band 1 | 60 MHz | 1 | ~350 ms |
| Band 3 | 80 MHz | 1 | ~420 ms |
| Band 28 | 50 MHz | 1 | ~300 ms |
| Band 8 | 40 MHz | 1 | ~280 ms |
| Band 40 | 100 MHz | 2 | ~554 ms (per log, 2x inference) |

Times are from the `tend - tstart` wall-clock measurement in `predict_samples()` which covers normalization + colormap + both model inferences + postprocessing. TCP overhead and protobuf serialization are not included.

**Note**: Apple M2 uses ARM architecture. OpenVINO CPU plugin on x86 production hardware (Intel/AMD) with AVX-512 VNNI may be 2–4x faster due to hardware-accelerated INT8 operations (though FP32 is used here). ARM NEON-accelerated paths in OpenVINO are also active on M2.

### 15.2 Startup Time

Model loading and warmup occurs before the first TCP connection is accepted. On a typical x86 server:
- 2G model load + compile: ~1–2 seconds
- 3G/4G model load + compile: ~0.5–1 second
- Warmup inferences: ~0.5 seconds
- Total startup: ~2–4 seconds

### 15.3 Memory Usage

Approximate runtime memory:
- OpenVINO runtime + CPU plugin: ~80 MB
- 2G model (FP32 weights in memory): ~99 MB
- 3G/4G model (FP32 weights in memory): ~21 MB
- Spectrogram processing arrays: ~10–50 MB (varies by bandwidth)
- Python interpreter + libraries: ~50 MB
- **Total**: ~260–300 MB RSS

---

## 17. Memory Management

### 16.1 Explicit Deletion Pattern

Scanner AI processes large NumPy arrays (spectrogram images can be tens of MB). Python's garbage collector does not immediately free large NumPy arrays because they use C-level memory that may not trigger GC thresholds. The code uses explicit `del` statements followed by `gc.collect()` to ensure memory is released promptly between requests:

```python
del blob                    # After inference
del colormapped_array       # After 3G/4G inference
del colormapped_array_2G    # After 2G inference
del detections_3g_4g, detections_2g
del data, spectrogram_new   # After full pipeline
gc.collect()               # Force immediate GC
```

This is especially important in a long-running TCP server that processes many requests: without explicit cleanup, memory usage would grow indefinitely until the OS kills the process.

### 16.2 Singleton Instances

The normalizer and colormap objects are created once:

```python
_normalizer = ai_colormap.NormalizePowerValue()
_colormap = ai_colormap.CustomImg()
```

The `CustomImg` instance holds a reference to the 256x3 viridis array. Creating a new instance per request would involve redundant re-reference of this array (small cost, but unnecessary).

### 16.3 Direct Float32 Blob Construction

In `preprocess_image`, the letterbox canvas is built directly in `float32`:

```python
blob = np.full((canvas_h, canvas_w, 3), 114.0 / 255.0, dtype=np.float32)
blob[...] = resized.astype(np.float32) * (1.0 / 255.0)
```

An alternative would be:
```python
canvas = np.full((canvas_h, canvas_w, 3), 114, dtype=np.uint8)
canvas[...] = resized
blob = canvas.astype(np.float32) / 255.0
```

The direct approach avoids the intermediate `uint8` canvas allocation (saving one full-sized array allocation) and avoids the extra division pass (the multiply by `1.0/255.0` is fused into the assignment).

---

## 18. Dependency Analysis

### 17.1 Runtime Dependencies (pyproject.toml)

```toml
dependencies = [
    "numpy>=1.24.0",
    "Pillow",
    "openvino>=2024.0.0",
    "opencv-python-headless>=4.8.0",
    "protobuf==3.20",
]
```

| Package | Purpose | Why this version |
|---------|---------|-----------------|
| `numpy>=1.24.0` | Array operations, FFT data handling | 1.24+ required for certain API compatibility with OpenVINO |
| `Pillow` | No direct use in current code (legacy dep) | May be needed for image I/O in edge cases |
| `openvino>=2024.0.0` | YOLO inference engine | 2024.0 introduced stable API and better dynamic shape support |
| `opencv-python-headless>=4.8.0` | `cv2.resize` for letterbox scaling | Headless variant avoids GUI dependencies in Docker |
| `protobuf==3.20` | TCP message serialization | Pinned to match `ai_model_pb2.py` generated code version |

### 17.2 Test Dependencies

```toml
[project.optional-dependencies]
test = ["coverage", "pytest"]
```

Not included in production image.

### 17.3 Export-Time Dependencies (Not in requirements.txt)

To run `export_openvino.py`, install separately:
```bash
pip install ultralytics   # Pulls PyTorch and all related dependencies
```

This is intentionally not in `requirements.txt` to keep the production dependency set minimal.

### 17.4 Dependency Comparison

| Dependency | Pre-migration (Ultralytics) | Post-migration (OpenVINO) |
|-----------|---------------------------|--------------------------|
| torch | Required (~800 MB) | Not needed |
| torchvision | Required (~200 MB) | Not needed |
| ultralytics | Required (~50 MB) | Not needed (export only) |
| openvino | Not needed | Required (~180 MB) |
| opencv-python | Required | Required (headless variant) |
| numpy | Required | Required |
| protobuf | Required | Required |
| **Total approx.** | **~1.1–1.3 GB** | **~250 MB** |

---

## 19. Design Decisions and Rationale

### 18.1 Why Two Separate Models Instead of One Multi-Class Model

A single model trained on all three classes (2G, 3G, 4G) would be the simpler approach. Two separate models are used because:

**Different signal scales**: 2G channels are 200 kHz wide while 4G channels are 5–20 MHz wide. A model must learn to detect both very narrow and very wide features simultaneously. Training two specialized models (one for narrow 2G, one for wider 3G/4G) produces better accuracy than a single model that must trade off between scales.

**Cascaded detection strategy**: 3G/4G signals are more reliably detected with high confidence. By detecting them first and masking them out, the 2G model receives a cleaner input (no 3G/4G dominating the spectrum) and can apply a lower confidence threshold (0.3 vs 0.6) without generating false positives from 3G/4G features.

**Different confidence thresholds**: If a single model were used, one confidence threshold would apply to all classes. The cascaded approach allows class-specific thresholds.

### 18.2 Why OpenVINO Over ONNX Runtime

Both OpenVINO and ONNX Runtime are viable alternatives to Ultralytics for CPU inference. OpenVINO was chosen because:

1. **Integration with Ultralytics export**: Ultralytics natively supports `model.export(format="openvino")`, producing ready-to-use IR files without manual ONNX intermediate steps.
2. **Intel-optimized CPU kernels**: OpenVINO's CPU plugin uses hand-tuned kernels for Intel architectures with AVX-512 and VNNI support, which covers most server hardware.
3. **LATENCY vs THROUGHPUT hints**: The `PERFORMANCE_HINT` API makes single-request latency optimization explicit without tuning thread counts manually.

### 18.3 Why Dynamic Shape Export

The `dynamic=True` export allows the model to accept inputs of variable dimensions. This is critical for the 2G model because:

The 2G inference receives a composite image assembled from the gaps between 3G/4G detections. The width of this image is `spectrogram_width - sum(3G/4G detection widths)`, which varies per request. The height equals the spectrogram row count, which depends on how many sweeps the hardware sent. Neither dimension is predictable at export time.

With `dynamic=True`, the model accepts any input size and the OpenVINO runtime adjusts execution accordingly. The `auto=True` letterbox mode then pads to the nearest stride multiple, minimizing unnecessary computation.

### 18.4 Why FP32 Over FP16

FP16 (half precision) would reduce model size by 50% and improve inference speed (especially on hardware with FP16 acceleration). FP32 is retained because:

The 2G model detects narrow 200 kHz signals that may appear at low power levels, just above the noise floor. The model has learned to distinguish these signals from noise via subtle activation patterns. FP16 rounding errors can collapse these fine-grained activations to zero. Empirical testing showed that FP16 quantization missed weak 2G detections that FP32 correctly identified.

For the 3G/4G model, FP16 would be acceptable, but FP32 is kept for simplicity (one export configuration for both models).

### 18.5 Why TCP Instead of HTTP/REST

The scanner hardware firmware communicates via raw TCP with a custom protobuf framing protocol. This was an existing protocol design constraint, not a new choice for v2. TCP was preferred by the hardware team because:

1. Lower latency than HTTP (no header parsing overhead)
2. Binary protobuf over raw TCP is more efficient than JSON over HTTP
3. The scanner device sends large binary payloads (float32 arrays) that map directly to TCP streaming without HTTP chunked encoding complexity

### 18.6 Why `num_khz_per_fft_point = 15`

The 30.72 MHz sample rate divided by 2048 FFT points gives exactly 15 kHz per FFT bin:

```
30,720,000 Hz / 2048 = 15,000 Hz = 15 kHz
```

30.72 MHz is the LTE chip rate (30.72 Mcps), chosen by 3GPP for LTE to be compatible with integer multiples of 3.84 Mcps (UMTS) and 13 MHz (GSM). Using the same sample rate for all three technologies allows a single hardware configuration to capture all signal types.

---

## 20. Troubleshooting Guide

### 19.1 Service Will Not Start

**Symptom**: `FileNotFoundError: No .xml file found in 2G_MODEL/best_openvino_model`

**Cause**: The model directory is missing or empty. The `.pt` file alone is not sufficient.

**Fix**: Run `export_openvino.py` to generate the OpenVINO IR files, or restore the `best_openvino_model/` directory from the repository.

---

**Symptom**: `ImportError: No module named 'openvino'`

**Cause**: OpenVINO is not installed.

**Fix**:
```bash
pip install openvino>=2024.0.0
# or
uv sync
```

---

**Symptom**: Server starts but immediately exits with no error

**Cause**: Socket bind failure (e.g., port 4444 already in use, `SO_REUSEADDR` not helping because a listening socket exists).

**Fix**: Find and kill the existing process:
```bash
lsof -ti:4444 | xargs kill -9
```

### 19.2 Inference Errors

**Symptom**: `Unexpected model output shape (X,)` logged, no detections

**Cause**: Model output tensor has unexpected dimensions, possibly because the model was exported with `end2end=True` (includes NMS in model).

**Fix**: Re-export with `end2end=False` (default). The metadata.yaml should show `end2end: false`.

---

**Symptom**: All coordinates returned as very small fractions (near 0.0)

**Cause**: Letterbox padding was not subtracted before coordinate inverse-transform. This indicates a mismatch between `auto=True` used in preprocessing vs `auto=False` used during result interpretation.

**Fix**: Ensure `auto` parameter is consistent between `preprocess_image` call and the model's expected behavior.

---

**Symptom**: 2G detections have large frequency errors (> 5 MHz off)

**Cause**: The `chunk_start_indexes_in_new_image` mapping is incorrect, likely due to a bug in the 2G composite image assembly.

**Debug**: Enable `SAVE_SAMPLES=YES` to save raw spectrogram images to `SAMPLES_LOW_POWER/`. Inspect the saved images to verify the composite assembly.

### 19.3 Wrong Frequencies Returned

**Symptom**: Detected frequencies are consistently shifted by a fixed amount

**Cause**: `start_freq` calculation uses wrong units. `center_freq` and `bandwidth` must both be in Hz when computing `start_freq = center_freq - bandwidth/2`.

**Verify**: The `predict_samples` function receives frequencies in kHz and immediately converts:
```python
center_freq_orig = center_freq_recv * emul  # kHz -> Hz? No: emul = 1e3, so kHz * 1000 = Hz
bandwidth = bandwidth_recv * emul           # same
```

Wait - check that `emul = 1e3` is correct for your unit system. The test parameters send `center_freq_khz` values like `2140000` (= 2140 MHz in kHz). After `* 1e3` this becomes `2.14e9` Hz = 2.14 GHz. Correct.

---

**Symptom**: Memory usage grows continuously over many requests

**Cause**: The `del` + `gc.collect()` pattern is not running (exception in `predict_samples` before cleanup).

**Fix**: Wrap the cleanup in a `finally` block, or check for exceptions swallowed upstream.

### 19.4 Test Failures

**Symptom**: Integration tests fail with "connection refused"

**Cause**: Server is not running, or is running on a different port.

**Fix**: Start `scanner.py` before running tests. Verify port 4444 is listening:
```bash
nc -z 127.0.0.1 4444 && echo "open" || echo "closed"
```

---

**Symptom**: Frequency not found within tolerance

**Cause**: Either the model accuracy has degraded (new model version), or the `FREQ_TOLERANCE_MHZ` constant is too tight.

**Diagnosis**: Add debug logging to print all detected frequencies before the assertion, and compare against expected values manually.

---

## 21. Appendices

### Appendix A: Model Directory Note

The `3G_4G_MODEL/best_openvino_model/` directory contains a nested `best_openvino_model/` subdirectory (visible in the directory listing). This is an artifact of an earlier export run that was not cleaned up. The production code `load_openvino_model("3G_4G_MODEL/best_openvino_model/")` finds the `.xml` at the top level and ignores the subdirectory. It should not cause issues but should be cleaned up to avoid confusion.

### Appendix B: Spectrogram Constants Reference

| Constant | Value | Meaning |
|----------|-------|---------|
| `fft_size` | 2048 | FFT bins per sweep |
| `num_khz_per_fft_point` | 15 | kHz per FFT bin |
| `fifteen_mhz_points` | 1000 | FFT bins for 15 MHz |
| `five_mhz_points` | 333 | FFT bins for 5 MHz |
| `MAX_AI_BANDWIDTH` | 60,000 (kHz) | Max inference band width |
| `edivide` | 1e6 | Hz -> MHz conversion |
| `emul` | 1e3 | kHz -> Hz conversion |
| Usable bins range | 357:1691 | In-band FFT bin window |
| Usable bin count | 1334 | Columns per sweep after trim |

### Appendix C: Protobuf Message Summary

```protobuf
// Approximate proto schema (ai_model_pb2.py is auto-generated)

enum AIResult {
    AI_RESULT_SUCCESS_UNSPECIFIED = 0;
    AI_RESULT_ERROR = 1;
}

message PredictSampleReq {
    int32 id = 1;
    float sampling_rate_khz = 2;
    float center_freq_khz = 3;
    float bw_khz = 4;
    int32 num_chunks = 5;
    float overlay_khz = 6;
    int32 samples_len = 7;
}

message SampleDataReq {
    int32 id = 1;
    repeated float samples = 2;
}

message AIModelReq {
    oneof message {
        PredictSampleReq predict_sample_req = 1;
        SampleDataReq sample_data_req = 2;
    }
}

message PredictSampleRes {
    AIResult result = 1;
    int32 id = 2;
}

message SampleDataRes {
    int32 id = 1;
    repeated float lte_freqs = 2;
    repeated float umts_freqs = 3;
    repeated float gsm_freqs = 4;
}

message AIModelRes {
    oneof message {
        PredictSampleRes predict_sample_res = 1;
        SampleDataRes sample_data_res = 2;
    }
}
```

### Appendix D: OpenVINO API Quick Reference

```python
import openvino as ov

# Create core (once per application)
core = ov.Core()

# Configure CPU performance
core.set_property("CPU", {
    "PERFORMANCE_HINT": "LATENCY",     # vs "THROUGHPUT"
    "INFERENCE_NUM_THREADS": "8",
})

# Load and compile model
model = core.read_model("model.xml")   # reads .xml + .bin automatically
compiled = core.compile_model(model, "CPU")

# Get I/O
input_layer = compiled.input(0)         # first input
output_layer = compiled.output(0)       # first output

# Query shape
partial_shape = input_layer.get_partial_shape()
is_static = partial_shape.is_static
shape = partial_shape.to_shape()        # Only if is_static

# Run inference
blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
result = compiled([blob])[output_layer]
# result shape: (1, 4+nc, N) for YOLO without end-to-end NMS
```

### Appendix E: Adding a New Model

To add a new signal technology model (e.g., 5G NR):

1. **Train YOLO model** on spectrogram images with 5G signal annotations
2. **Export to OpenVINO**:
   ```python
   # In export_openvino.py, add to MODEL_DIRS
   MODEL_DIRS = ["2G_MODEL", "3G_4G_MODEL", "5G_MODEL"]
   ```
3. **Load model at startup** in `scanner.py`:
   ```python
   model_5g_compiled, model_5g_output, model_5g_shape = load_openvino_model(
       "5G_MODEL/best_openvino_model/")
   model_5g_h, model_5g_w = int(model_5g_shape[2]), int(model_5g_shape[3])
   ```
4. **Add inference call** in `predict_samples()` after 3G/4G stage
5. **Add `nr_freqs` to response** in `recieve_samples()` and update the protobuf schema
6. **Add test expectations** in `test_scanner_ai_script.py`
7. **Update Dockerfile** to copy the new model directory

### Appendix F: Glossary

| Term | Definition |
|------|-----------|
| FFT | Fast Fourier Transform - converts time-domain signal to frequency-domain power spectrum |
| dBm | Decibels relative to 1 milliwatt - unit for RF power measurement |
| Spectrogram | 2D representation of spectrum over time (frequency vs time, color = power) |
| Letterbox | Image resizing technique that maintains aspect ratio by adding gray padding |
| NMS | Non-Maximum Suppression - removes duplicate overlapping bounding boxes |
| IOU | Intersection-over-Union - measure of bounding box overlap |
| IR | Intermediate Representation - OpenVINO's model format (.xml + .bin) |
| FP32 | 32-bit floating point precision |
| INT8 | 8-bit integer quantization (4x smaller than FP32, with accuracy tradeoff) |
| YOLO | You Only Look Once - single-pass object detection architecture |
| GSM | Global System for Mobile (2G technology) |
| UMTS | Universal Mobile Telecommunications System (3G technology) |
| LTE | Long Term Evolution (4G technology) |
| FDD | Frequency Division Duplex - uplink and downlink on separate frequency bands |
| TDD | Time Division Duplex - uplink and downlink share same frequency, separated by time |
| Center Frequency | The middle frequency of a signal's channel bandwidth |
| Bandwidth | Total width of a frequency band in Hz/MHz |

---

*Document generated: 2026-02-21*
*Repository: https://github.com/Manikanta-Reddy-Pasala/UltralyticsToOpenvino*
*Primary source file: `/Users/manip/Documents/codeRepo/scanner_ai_v2/scanner.py`*
