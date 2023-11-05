import onnxruntime as ort
import cv2
import numpy as np
import time

from utils.infer import PredictConfig,  classify_image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer_cfg", type=str, default="./clas/infer_cfg.yml", help="infer_cfg.yml")
    parser.add_argument(
        '--onnx_file', type=str, default="./clas/4clas_trash_MobileNetV3_x_1_0_95_7.onnx", help="onnx model file path")
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--image_file", type=str, default="")

    # Load the ONNX model
    FLAGS = parser.parse_args()

        # load predictor
    predictor = ort.InferenceSession(FLAGS.onnx_file)
    # load infer config
    infer_config = PredictConfig(FLAGS.infer_cfg)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame,1)
        t1 = time.time()

        clas, prob = classify_image(infer_config, predictor, frame)
        print("FPS:" , 1/(time.time() - t1))
        
        frame = cv2.putText(frame, str(clas) + " :" + str(prob), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255),1)
        cv2.imshow("image", frame)
        cv2.waitKey(1)