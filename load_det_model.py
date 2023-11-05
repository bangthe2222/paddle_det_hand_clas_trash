import onnxruntime as ort
import cv2
import numpy as np
import time

from utils.infer import PredictConfig, predict_image
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer_cfg", type=str, default="./det/infer_cfg.yml", help="infer_cfg.yml")
    parser.add_argument(
        '--onnx_file', type=str, default="./det/hand_object_detect_picodet_s_320.onnx", help="onnx model file path")
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--image_file", type=str, default="")

    # Load the ONNX model
    FLAGS = parser.parse_args()

        # load predictor
    predictor = ort.InferenceSession(FLAGS.onnx_file, providers=["CUDAExecutionProvider"])
    # load infer config
    infer_config = PredictConfig(FLAGS.infer_cfg)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        height, width = frame.shape[:2]
        print(height, width)
        t1 = time.time()
        boxes = predict_image(infer_config, predictor, frame, im_show=True)
        # print(boxes)
        for box in boxes:
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
            w0, h0 = int((x1-  x0)/2), int((y1 - y0)/2)
            
            y_min = (y0 - h0) if (y0 - h0) >0 else 0
            y_max = (y1 + h0) if (y1 + h0) < height else height
            x_min = (x0 - h0) if (x0 - h0) >0 else 0
            x_max = (x1 + h0) if (x1 + h0) < width else width
            img_crop = frame[y_min:y_max,x_min:x_max]
            
        print("FPS:" , 1/(time.time() - t1))