import numpy as np
import socket
from google.protobuf import message
import os
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ai_model_pb2

Bands_to_be_tested = [1,3,8,20,28,40]

#Folder path to the samples for testing
folder_dir = "SAMPLES_UT"

expected_4g_frequencies = {
        "Band_1"    : [2165.0, 2146.7],
        "Band_3"    : [1815.0, 1870.0, 1849.5],
        "Band_8"    : [],
        "Band_20"   : [813.6, 798.5],
        "Band_28"   : [763.1, 800.8],
        "Band_40"   : [2342.1, 2361.9]
        }

expected_3g_frequencies = {
        "Band_1"    : [2116.4, 2137.7],
        "Band_3"    : [],
        "Band_8"    : [932.6, 937.2, 927.5],
        "Band_20"   : [],
        "Band_28"   : [],
        "Band_40"   : []
        }


expected_2g_frequencies = {
        "Band_1"    : [],
        "Band_3"    : [1860.2],
        "Band_8"    : [953.4],
        "Band_20"   : [],
        "Band_28"   : [],
        "Band_40"   : []
        }

bandwise_parameters = {
        "Band_1"    :  [2140000 , 60000  , 5],# Center Freq, Bandwidth, Num Chunks
        "Band_3"    :  [1845000 , 80000  , 7],
        "Band_8"    :  [945000  , 40000  , 3],
        "Band_20"   :  [806000  , 30000  , 2],
        "Band_28"   :  [783000  , 50000  , 4],
        "Band_40"   :  [2350000 , 100000 , 9]
        }

def samples_test(file,band):
    detected_4g_freq = []
    detected_3g_freq = []
    detected_2g_freq = []
    client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1',4444))
    send_buffer = np.fromfile(folder_dir+ '/' + file,dtype=np.float32)

    scanner_ai_predict_data_req = ai_model_pb2.AIModelReq()
    scanner_ai_predict_data_req.sample_data_req.id = band
    scanner_ai_predict_data_req.sample_data_req.samples.extend(send_buffer)
    data_req = scanner_ai_predict_data_req.SerializeToString()

    """ Prepare the the predict sample req for AI module with hyperparameters"""
    scanner_ai_predict_sample_req = ai_model_pb2.AIModelReq()

    scanner_ai_predict_sample_req.predict_sample_req.id			   = band
    scanner_ai_predict_sample_req.predict_sample_req.sampling_rate_khz     = 30720
    scanner_ai_predict_sample_req.predict_sample_req.center_freq_khz       = bandwise_parameters[f"Band_{band}"][0]
    scanner_ai_predict_sample_req.predict_sample_req.bw_khz                = bandwise_parameters[f"Band_{band}"][1]
    scanner_ai_predict_sample_req.predict_sample_req.num_chunks            = bandwise_parameters[f"Band_{band}"][2]
    scanner_ai_predict_sample_req.predict_sample_req.overlay_khz           = 10000
    scanner_ai_predict_sample_req.predict_sample_req.samples_len           = len(data_req)

    sample_req = scanner_ai_predict_sample_req.SerializeToString()

    """ Send the predict sample req to the AI moudle """
    print(f"Sending the sample...")
    client_socket.send(sample_req)

    """Recieve the ACK from the AI module"""

    buf = client_socket.recv(50)
    ai_model_ack = ai_model_pb2.AIModelRes()
    ai_model_ack.ParseFromString(buf)

    if ai_model_ack.WhichOneof("message") == "init_res":
        print(f"Deprecated Message Init")
    elif ai_model_ack.WhichOneof("message") == "predict_sample_res":

        if ai_model_ack.predict_sample_res.result == ai_model_pb2.AIResult.AI_RESULT_SUCCESS_UNSPECIFIED and ai_model_ack.predict_sample_res.id == band:
            print(f"Received Ack for Predict Sample req message")
        elif ai_model_ack.predict_sample_res.result == ai_model_pb2.AIResult.AI_RESULT_SUCCESS_UNSPECIFIED and ai_model_ack.predict_sample_res.id != band:
            print(f"Received Ack for wrong band request sent for {band} received respons for {ai_model_ack.predict_sample_res.id}")
            return detected_4g_freq, detected_3g_freq, detected_2g_freq
        else:
            print(f"Recieved Error for predict sample request for Band {band}")
            return detected_4g_freq, detected_3g_freq, detected_2g_freq

    else:
        print(f"ERROR : Expected predict_sample_res Receieved sample_data_res")

    client_socket.sendall(data_req)

    """Recieve the predictions from the AI module """
    buf = client_socket.recv(1024)
    ai_model_predictions = ai_model_pb2.AIModelRes()
    ai_model_predictions.ParseFromString(buf)

    if ai_model_predictions.WhichOneof("message") == "init_res":
        print(f"Deprecated Message Init")

    elif ai_model_predictions.WhichOneof("message") == "predict_sample_res":
        print(f"ERROR Expected sample_data_res Recieved predict_sample_res")

    elif ai_model_predictions.WhichOneof("message") == "sample_data_res":

        if ai_model_predictions.sample_data_res.id == band:
            print(f"Recieved sample_data_res from AI module with id {ai_model_predictions.sample_data_res.id}")
            print(f"Predicted 4G frequencies : {ai_model_predictions.sample_data_res.lte_freqs}")
            detected_4g_freq = ai_model_predictions.sample_data_res.lte_freqs
            print(f"Predicted 3G frequencies : {ai_model_predictions.sample_data_res.umts_freqs}")
            detected_3g_freq = ai_model_predictions.sample_data_res.umts_freqs
            print(f"Predicted 2G frequencies : {ai_model_predictions.sample_data_res.gsm_freqs }")
            detected_2g_freq = ai_model_predictions.sample_data_res.gsm_freqs
        else:
            print(f"Received sample data response with incorrect id {ai_model_predictions.sample_data_res.id}")
    else:
        print(f"Wrong message received")

    client_socket.close()
    return detected_4g_freq, detected_3g_freq, detected_2g_freq


def test_check_detected_freq():
    for bands in Bands_to_be_tested:
        file = f"sample_vec_B{bands}.dat"
        detected_4g, detected_3g, detected_2g = samples_test(file,bands)
        detected_4g = [round(x, 1) for x in detected_4g]
        detected_3g = [round(x, 1) for x in detected_3g]
        detected_2g = [round(x, 1) for x in detected_2g]
        assert  set(expected_4g_frequencies[f"Band_{bands}"]).issubset(detected_4g)
        assert  set(expected_3g_frequencies[f"Band_{bands}"]).issubset(detected_3g)
        assert  set(expected_2g_frequencies[f"Band_{bands}"]).issubset(detected_2g)
