import os
import logging
import numpy as np
import cv2
import openvino as ov
import ai_colormap
from datetime import datetime
import socket
import ai_model_pb2
from google.protobuf import message
import threading
from scanner_logging import setup_logging
import gc
import signal
import sys


setup_logging()

###################PARAMETERS FROM THE HEADER##################################################
sample_rate = 30.72e6
center_freq = 938.9e6 #4G
bandwidth = 58.6e6
num_center_frequencies = 3
overlap = 1
fft_size = 2048
num_khz_per_fft_point = 15
fifteen_mhz_points = int(15000/num_khz_per_fft_point)
five_mhz_points = int(5000/num_khz_per_fft_point)
predicted_freq_list_2g = []
predicted_freq_list_3g = []
predicted_freq_list_4g = []


################################ OpenVINO INFERENCE ENGINE #####################################

def _read_model_imgsz(model_dir):
    """Read imgsz from metadata.yaml in the model directory (simple parser, no PyYAML needed)."""
    meta_path = os.path.join(model_dir, "metadata.yaml")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'r') as f:
        lines = f.readlines()
    # Find 'imgsz:' followed by '- NNN' lines
    imgsz = []
    in_imgsz = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("imgsz:"):
            in_imgsz = True
            continue
        if in_imgsz:
            if stripped.startswith("- "):
                try:
                    imgsz.append(int(stripped[2:]))
                except ValueError:
                    break
            else:
                break
    if len(imgsz) == 2:
        return imgsz
    return None


def load_openvino_model(model_dir, num_threads=4):
    """Load an OpenVINO model from a directory containing .xml and .bin files."""
    core = ov.Core()
    core.set_property("CPU", {
        "INFERENCE_NUM_THREADS": str(num_threads),
        "PERFORMANCE_HINT": "LATENCY",
    })
    xml_files = [f for f in os.listdir(model_dir) if f.endswith(".xml")]
    if not xml_files:
        raise FileNotFoundError(f"No .xml file found in {model_dir}")
    model_file = os.path.join(model_dir, xml_files[0])
    model = core.read_model(model_file)
    compiled_model = core.compile_model(model, "CPU")
    output_layer = compiled_model.output(0)
    partial_shape = compiled_model.input(0).get_partial_shape()
    if partial_shape.is_static:
        input_shape = list(partial_shape.to_shape())
    else:
        # Dynamic shape - read from metadata.yaml
        meta_imgsz = _read_model_imgsz(model_dir)
        if meta_imgsz:
            input_shape = [1, 3, meta_imgsz[0], meta_imgsz[1]]
            logging.info(f"Dynamic shape model, using imgsz from metadata: {meta_imgsz}")
        else:
            input_shape = [1, 3, 640, 640]
            logging.info(f"Dynamic shape model, no metadata found, using default 640x640")
    logging.info(f"Loaded OpenVINO model: {model_file}, input shape: {input_shape}")
    return compiled_model, output_layer, input_shape


def preprocess_image(img, target_h, target_w, auto=False, stride=32):
    """Letterbox preprocess BGR uint8 image for YOLO OpenVINO inference.

    Args:
        img: BGR uint8 numpy array (H, W, 3)
        target_h, target_w: model's default input dimensions
        auto: if True, use stride-aligned minimal padding (like ultralytics .pt inference)
        stride: alignment stride when auto=True

    Returns:
        blob: float32 (1, 3, canvas_h, canvas_w)
        letterbox_info: dict with scale and padding info for coordinate mapping
    """
    orig_h, orig_w = img.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    if auto:
        # Minimal stride-aligned padding (matches ultralytics auto=True behavior)
        pad_w = (stride - new_w % stride) % stride
        pad_h = (stride - new_h % stride) % stride
        canvas_w = new_w + pad_w
        canvas_h = new_h + pad_h
        # Center padding
        pad_left = pad_w // 2
        pad_top = pad_h // 2
    else:
        canvas_w = target_w
        canvas_h = target_h
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((canvas_h, canvas_w, 3), 114, dtype=np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[::-1]  # HWC->CHW and BGR->RGB (matches ultralytics)
    blob = np.ascontiguousarray(blob)
    blob = np.expand_dims(blob, 0)    # add batch dimension
    letterbox_info = {'scale': scale, 'pad_w': pad_left, 'pad_h': pad_top,
                      'orig_w': orig_w, 'orig_h': orig_h}
    return blob, letterbox_info


def numpy_nms(x1, y1, x2, y2, scores, iou_threshold):
    """Non-Maximum Suppression (pure numpy)."""
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
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


def postprocess_yolo_output(output, conf_threshold, letterbox_info, iou_threshold=0.45):
    """Parse raw YOLO output tensor into detections.

    Args:
        output: shape (1, 4+num_classes, N) - raw model output
        conf_threshold: minimum confidence
        letterbox_info: dict from preprocess_image
        iou_threshold: NMS IoU threshold

    Returns:
        list of dicts: {'xywh': (cx, cy, w, h), 'cls': int, 'conf': float}
        coordinates are in original image pixel space
    """
    raw = output[0]  # (4+nc, N)
    if raw.ndim != 2 or raw.shape[0] <= 4:
        logging.warning(f"Unexpected model output shape {raw.shape}")
        return []
    predictions = raw.T  # (N, 4+nc)

    class_scores = predictions[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    mask = max_scores > conf_threshold
    predictions = predictions[mask]
    max_scores = max_scores[mask]

    if len(predictions) == 0:
        return []

    class_ids = np.argmax(predictions[:, 4:], axis=1)
    boxes = predictions[:, :4].copy()

    # Remove letterbox padding and rescale to original image coords
    scale = letterbox_info['scale']
    pad_w = letterbox_info['pad_w']
    pad_h = letterbox_info['pad_h']
    boxes[:, 0] = (boxes[:, 0] - pad_w) / scale  # cx
    boxes[:, 1] = (boxes[:, 1] - pad_h) / scale  # cy
    boxes[:, 2] = boxes[:, 2] / scale              # w
    boxes[:, 3] = boxes[:, 3] / scale              # h

    # Convert to xyxy for NMS
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # Per-class NMS (matches ultralytics default behavior)
    num_classes = int(class_scores.shape[1]) if class_scores.ndim == 2 else 1
    all_keep = []
    for cls_id in range(num_classes):
        cls_mask = class_ids == cls_id
        if not np.any(cls_mask):
            continue
        cls_indices = np.where(cls_mask)[0]
        cls_keep = numpy_nms(x1[cls_indices], y1[cls_indices],
                             x2[cls_indices], y2[cls_indices],
                             max_scores[cls_indices], iou_threshold)
        all_keep.extend(cls_indices[k] for k in cls_keep)

    detections = []
    for i in all_keep:
        detections.append({
            'xywh': (float(boxes[i, 0]), float(boxes[i, 1]),
                     float(boxes[i, 2]), float(boxes[i, 3])),
            'cls': int(class_ids[i]),
            'conf': float(max_scores[i]),
        })

    return detections


def run_inference(compiled_model, output_layer, img, conf_threshold, target_h, target_w, iou_threshold=0.45, auto=False):
    """Full inference pipeline: preprocess -> infer -> postprocess."""
    blob, letterbox_info = preprocess_image(img, target_h, target_w, auto=auto)
    raw_output = compiled_model([blob])[output_layer]
    detections = postprocess_yolo_output(raw_output, conf_threshold, letterbox_info, iou_threshold)
    logging.info(f"Inference: img {img.shape} -> {blob.shape[2]}x{blob.shape[3]}, detections: {len(detections)}")
    return detections


################################AI MODEL IMPORT#################################################
model_2g_compiled, model_2g_output, model_2g_shape = load_openvino_model("2G_MODEL/best_int8_openvino_model/")
model_3g_4g_compiled, model_3g_4g_output, model_3g_4g_shape = load_openvino_model("3G_4G_MODEL/best_openvino_model/")

# Extract target sizes from model input shapes: [1, 3, H, W]
model_2g_h, model_2g_w = int(model_2g_shape[2]), int(model_2g_shape[3])
model_3g_4g_h, model_3g_4g_w = int(model_3g_4g_shape[2]), int(model_3g_4g_shape[3])

edivide = 1e6
emul = 1e3
MAX_AI_BANDWIDTH = 60 * emul


def sigterm_handler(_signo, _stack_frame):
    logging.info("Received SIGTERM, exiting...")
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)

###################################WARMUP FOR MODEL############################################
logging.info("Warming up 2G model...")
dummy_blob = np.zeros((1, 3, model_2g_h, model_2g_w), dtype=np.float32)
model_2g_compiled([dummy_blob])[model_2g_output]

logging.info("Warming up 3G/4G model...")
dummy_blob = np.zeros((1, 3, model_3g_4g_h, model_3g_4g_w), dtype=np.float32)
model_3g_4g_compiled([dummy_blob])[model_3g_4g_output]
del dummy_blob

##################################CREATE SOCKET ##############################################
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def add_detected_cells_to_list(predicted_center_freq,clsitem,is_3g_4g,xval_list,xval,width):

    if is_3g_4g == True:
        if clsitem == 0:
            xval_list.append([int(xval - width / 2) ,int(width)])
            if predicted_center_freq not in predicted_freq_list_3g:
                predicted_freq_list_3g.append(predicted_center_freq)
        elif clsitem == 1 or clsitem == 2:
            xval_list.append([int(xval - width / 2) ,int(width)])
            if predicted_center_freq not in predicted_freq_list_4g:
                predicted_freq_list_4g.append(predicted_center_freq)
    else:
        if predicted_center_freq not in predicted_freq_list_2g:
            predicted_freq_list_2g.append(predicted_center_freq)



def process_results_3g_4g(detections_3g_4g,xval_list,start_freq,bandwidth,img_width):
    for det in detections_3g_4g:
        clsitem = det['cls']
        cx, cy, w, h = det['xywh']
        xval = cx
        width = w
        pixel_size_of_img = img_width
        predicted_center_freq = ((bandwidth/pixel_size_of_img) * xval)
        predicted_center_freq += start_freq
        predicted_center_freq /= edivide
        predicted_center_freq= round(float(predicted_center_freq), 1)
        add_detected_cells_to_list(predicted_center_freq,
                clsitem,
                True,
                xval_list,
                xval,
                width)
    xval_list.sort()
    return xval_list


def process_results_2g(detections_2g,start_freq_for_each_chunk,chunk_start_indexes_in_new_image):
    index_val = 0
    for det in detections_2g:
        clsitem = det['cls']
        cx, cy, w, h = det['xywh']
        x_val_in_image = cx
        i = 0
        while i < len(chunk_start_indexes_in_new_image):
           if x_val_in_image < chunk_start_indexes_in_new_image[i]:
               index_val = i - 1
           else:
               index_val = i
           i = i + 1

        predicted_center_freq = (15000 * (x_val_in_image - chunk_start_indexes_in_new_image[index_val]))
        start_freq_chunk_wise = start_freq_for_each_chunk[index_val]

        predicted_center_freq += start_freq_chunk_wise
        predicted_center_freq /= edivide
        predicted_center_freq= round(float(predicted_center_freq), 1)
        add_detected_cells_to_list(predicted_center_freq,
                clsitem,
                False,
                None,
                0,
                0)


def get_num_chunks_for_mem_optimization(bandwidth_recv,overlap):
    num_chunks = 1
    if bandwidth_recv >= 100 * emul:
        """
        12 MHZ overlap to bifurcate the spectrum to reduce the memory consumption in case of large bands exceeding 100 Mhz
        """
        overlap = int(12 * emul / num_khz_per_fft_point)
        # Here use +1 when the divison with 48 is significantly bigger to make an impact on the last chunk
        num_chunks = int(bandwidth_recv // (MAX_AI_BANDWIDTH  - overlap * num_khz_per_fft_point))
    return num_chunks,overlap


def create_correct_spectrogram_by_rearranging_samples(num_center_frequencies,spectrogram,num_of_samples_in_freq):
    loop_counter = 1
    if num_center_frequencies == 2:
        spectrogram_new = np.append(spectrogram[0:num_of_samples_in_freq - 1 , : fifteen_mhz_points], spectrogram[num_of_samples_in_freq * loop_counter: (num_of_samples_in_freq * (loop_counter + 1)) - 1 ,five_mhz_points :] ,axis=1)

    elif num_center_frequencies == 1:
        spectrogram_new = spectrogram

    else:
       while loop_counter <= num_center_frequencies - 1:
           if loop_counter == 1:
               spectrogram_new = np.append(spectrogram[0:num_of_samples_in_freq - 1 , : fifteen_mhz_points], spectrogram[num_of_samples_in_freq * loop_counter: (num_of_samples_in_freq * (loop_counter + 1)) - 1 ,five_mhz_points : fifteen_mhz_points] ,axis=1)

           elif loop_counter == num_center_frequencies-1:
               spectrogram_new = np.append(spectrogram_new[0:num_of_samples_in_freq - 1,:] , spectrogram[num_of_samples_in_freq * loop_counter: (num_of_samples_in_freq * (loop_counter + 1)) - 1 , five_mhz_points :], axis=1)

           else:
               spectrogram_new = np.append(spectrogram_new[0:num_of_samples_in_freq - 1,:],spectrogram[num_of_samples_in_freq * loop_counter: (num_of_samples_in_freq * (loop_counter + 1)) - 1, five_mhz_points : fifteen_mhz_points],axis=1)

           loop_counter += 1
    return spectrogram_new


def get_truncated_spectrum(spectrogram_new,num_chunks,chunk_iterator,center_freq_orig,num_samples_in_chunk,bandwidth,overlap):

    if num_chunks > 1:

        if chunk_iterator != 0:
            start_index = (chunk_iterator * num_samples_in_chunk) - (overlap * chunk_iterator)
            end_index = start_index + num_samples_in_chunk
            if end_index < spectrogram_new.shape[1]:
                spectrogram_predict = spectrogram_new[:,start_index : end_index]
                center_freq = center_freq_orig + int(MAX_AI_BANDWIDTH) * emul - overlap * num_khz_per_fft_point * emul
                bandwidth = MAX_AI_BANDWIDTH * emul
                center_freq_orig = center_freq
            else:
                end_index = spectrogram_new.shape[1]
                spectrogram_predict = spectrogram_new[:,start_index : end_index]
                bandwidth = (end_index - start_index) * num_khz_per_fft_point
                center_freq = center_freq_orig + int(MAX_AI_BANDWIDTH/2) * emul + int(bandwidth/2) * emul - overlap * num_khz_per_fft_point * emul
                bandwidth = bandwidth * emul
                center_freq_orig = center_freq

        else:
            spectrogram_predict = spectrogram_new[:,0 : num_samples_in_chunk]
            # center freq of the signal or the first chunk
            center_freq = (center_freq_orig  - int((bandwidth / 2))) + (int((MAX_AI_BANDWIDTH / 2)) * emul )
            bandwidth = MAX_AI_BANDWIDTH * emul
            center_freq_orig = center_freq
    else:
        spectrogram_predict = None
        center_freq = None

    return spectrogram_predict, center_freq, bandwidth, center_freq_orig

def save_sample(colormapped_array, center_freq):
    cv2.imwrite('SAMPLES_LOW_POWER/SCANNER_SAMPLES_CF_'+str(center_freq) + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.jpg',colormapped_array)

def predict_samples(center_freq_recv,bandwidth_recv,num_center_frequencies_recv,overlap_recv,data,save_samples,memory_optimization):
    predicted_freq_list_4g.clear()
    predicted_freq_list_3g.clear()
    predicted_freq_list_2g.clear()
    center_freq_orig = center_freq_recv * emul
    bandwidth = bandwidth_recv * emul
    num_center_frequencies = num_center_frequencies_recv
    overlap = overlap_recv * emul

    tstart = datetime.now()
    spectrogram = np.copy(data)
    num_of_samples_in_freq = spectrogram.shape[0]//num_center_frequencies

    spectrogram =  spectrogram[:,357:]
    spectrogram =  spectrogram[:,0:1334]

    num_chunks = 1

    spectrogram_new = create_correct_spectrogram_by_rearranging_samples(num_center_frequencies,spectrogram,num_of_samples_in_freq)

    if memory_optimization == "YES":
        num_chunks , overlap = get_num_chunks_for_mem_optimization(bandwidth_recv,overlap)

    num_samples_in_chunk = int(MAX_AI_BANDWIDTH/num_khz_per_fft_point)


    for chunk_iterator in range(num_chunks):

        spectrogram_predict, center_freq, bandwidth, center_freq_orig  = get_truncated_spectrum(spectrogram_new,
                num_chunks,
                chunk_iterator,
                center_freq_orig,
                num_samples_in_chunk,
                bandwidth,
                overlap)

        n = ai_colormap.NormalizePowerValue()

        if memory_optimization == "YES" and num_chunks > 1:
            img = n.get_normalized_values(spectrogram_predict)
        else:
            img = n.get_normalized_values(spectrogram_new)
            center_freq = center_freq_orig

        color_obj = ai_colormap.CustomImg()
        colormapped_array = color_obj.get_new_img(img)
        colormapped_array = colormapped_array.astype(np.uint8)
        colormapped_array = colormapped_array[...,::-1]
        del img
        xval_list = []
        chunk_start_indexes_in_new_image = []
        start_freq_for_each_chunk = []
        start_freq = center_freq - (bandwidth/2)

        if save_samples == "YES":
            save_sample(colormapped_array, center_freq)

        # 3G/4G inference via OpenVINO
        detections_3g_4g = run_inference(model_3g_4g_compiled, model_3g_4g_output,
                                         colormapped_array, conf_threshold=0.6,
                                         target_h=model_3g_4g_h, target_w=model_3g_4g_w,
                                         auto=True)

        img_width = colormapped_array.shape[1]
        xval_list = process_results_3g_4g(detections_3g_4g,xval_list,start_freq,bandwidth,img_width)

        for i in range(len(xval_list)):
            if i == 0 :
                colormapped_array_2G = colormapped_array[:,0:xval_list[i][0],:]
                logging.info(f"Colormapped array new shape {colormapped_array_2G.shape}")
                chunk_start_indexes_in_new_image.append(0)
                start_freq_for_each_chunk.append(start_freq)
            else:
                start_freq_for_each_chunk.append(start_freq + (xval_list[i-1][0] + xval_list[i-1][1]) * 15000)
                chunk_start_indexes_in_new_image.append(colormapped_array_2G.shape[1])
                colormapped_array_2G = np.concatenate((colormapped_array_2G,colormapped_array[: , xval_list[i-1][0] + xval_list[i-1][1] : xval_list[i][0] , :]),axis=1)

        if len(xval_list) != 0:
            chunk_start_indexes_in_new_image.append(colormapped_array_2G.shape[1])
            colormapped_array_2G = np.concatenate((colormapped_array_2G,colormapped_array[: , xval_list[i][0] + xval_list[i][1] : , :]),axis=1)
            start_freq_for_each_chunk.append(start_freq + (xval_list[i][0] + xval_list[i][1]) * 15000)
        else:
            colormapped_array_2G = colormapped_array
            chunk_start_indexes_in_new_image.append(0)
            start_freq_for_each_chunk.append(start_freq)

        # 2G inference via OpenVINO
        # Original code passed imgsz=[img_h, img_w] (image's own size), so no resize, just stride-align
        detections_2g = run_inference(model_2g_compiled, model_2g_output,
                                      colormapped_array_2G, conf_threshold=0.3,
                                      target_h=colormapped_array_2G.shape[0],
                                      target_w=colormapped_array_2G.shape[1],
                                      auto=True)

        process_results_2g(detections_2g,start_freq_for_each_chunk,chunk_start_indexes_in_new_image)

        tend = datetime.now()
    del spectrogram,data,colormapped_array_2G,detections_3g_4g,detections_2g
    gc.collect()


    return (list(predicted_freq_list_4g),list(predicted_freq_list_3g),list(predicted_freq_list_2g),tend-tstart)

def recieve_samples(conn,initial_byte_size,scanner_ai_save_samples,memory_optimization):
        buf = conn.recv(initial_byte_size)
        scanner_ai_data_req = ai_model_pb2.AIModelReq()

        try:
            scanner_ai_data_req.ParseFromString(buf)
        except  message.DecodeError as e:
            logging.error(f"Unable to parse incoming message: {buf}{e}")
            return

        scanner_ai_res = ai_model_pb2.AIModelRes()
        scanner_ai_res.predict_sample_res.result = ai_model_pb2.AIResult.AI_RESULT_SUCCESS_UNSPECIFIED
        scanner_ai_res.predict_sample_res.id = scanner_ai_data_req.predict_sample_req.id


        payload = scanner_ai_res.SerializeToString()
        conn.send(payload)

        if scanner_ai_data_req.WhichOneof("message") == "predict_sample_req":
            sample_rate = scanner_ai_data_req.predict_sample_req.sampling_rate_khz
            center_freq = scanner_ai_data_req.predict_sample_req.center_freq_khz
            bandwidth = scanner_ai_data_req.predict_sample_req.bw_khz
            num_center_freq = scanner_ai_data_req.predict_sample_req.num_chunks
            overlap = scanner_ai_data_req.predict_sample_req.overlay_khz
            sample_len = scanner_ai_data_req.predict_sample_req.samples_len

        else:
            logging.error("Wrong message type received expected predict_sample_req")
            return

        bytes_recd = 0
        chunks = []

        while bytes_recd < sample_len:
            chunk = conn.recv(min(sample_len - bytes_recd, 65000))
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)

        scanner_ai_data_req_1 = ai_model_pb2.AIModelReq()
        scanner_ai_data_req_1.ParseFromString(b''.join(chunks))

        if scanner_ai_data_req_1.WhichOneof("message") == "sample_data_req":
            sample_data_req_id = scanner_ai_data_req_1.sample_data_req.id
            sample = np.array(scanner_ai_data_req_1.sample_data_req.samples,dtype=np.float32)
            length = len(sample)
            sample = sample.reshape(length//fft_size,fft_size)
            predicted_4g, predicted_3g, predicted_2g, time_taken = predict_samples(center_freq,bandwidth,num_center_freq,overlap,sample,scanner_ai_save_samples,memory_optimization)
        else:
            logging.error("Wrong message type received expected sample_data_req")

        logging.info(f"predicted 4G {predicted_4g}")
        scanner_ai_data_res = ai_model_pb2.AIModelRes()

        scanner_ai_data_res.sample_data_res.id = sample_data_req_id
        scanner_ai_data_res.sample_data_res.lte_freqs[:] = predicted_4g[:]
        scanner_ai_data_res.sample_data_res.umts_freqs[:] = predicted_3g[:]
        scanner_ai_data_res.sample_data_res.gsm_freqs[:] = predicted_2g[:]
        conn.send(scanner_ai_data_res.SerializeToString())

        logging.info("------------------HEADER CONTENTS-----------------")
        logging.info(f"sample rate {scanner_ai_data_req.predict_sample_req.sampling_rate_khz}")
        logging.info(f"center freq {scanner_ai_data_req.predict_sample_req.center_freq_khz}")
        logging.info(f"bandwidth {scanner_ai_data_req.predict_sample_req.bw_khz}")
        logging.info(f"chunks {scanner_ai_data_req.predict_sample_req.num_chunks}")
        logging.info(f"overlap {scanner_ai_data_req.predict_sample_req.overlay_khz}")
        logging.info(f"sample len {scanner_ai_data_req.predict_sample_req.samples_len}")

        logging.info("------------------AI MODEL PREDICTIONS-----------------")
        logging.info(f"Detected 4G frequencies by AI {predicted_4g}")
        logging.info(f"Detected 3G frequencies by AI {predicted_3g}")
        logging.info(f"Detected 2G frequencies by AI {predicted_2g}")


        logging.info(f"****************Total time taken by AI MODELS ****************************** {time_taken}")
        del sample,scanner_ai_data_req_1
        del chunks
        del predicted_4g, predicted_3g, predicted_2g
        gc.collect()

def main():

    scanner_ai_host = os.getenv('SCANNER_AI_IP', '0.0.0.0')
    scanner_ai_port = int(os.getenv('SCANNER_AI_PORT', 4444))
    scanner_ai_save_samples = str(os.getenv('SAVE_SAMPLES'))
    memory_optimization = str(os.getenv('MEM_OPTIMIZATION'))
    logging.info(f"Starting Scanner AI service listening on port {scanner_ai_port} {scanner_ai_host}...")
    s.bind((scanner_ai_host, scanner_ai_port))
    s.listen(1)
    while True:
        connection, _ = s.accept()
        reciever_thread = threading.Thread(target=recieve_samples, args= (connection,1024,scanner_ai_save_samples,memory_optimization))
        reciever_thread.start()

if __name__=="__main__":
    main()
