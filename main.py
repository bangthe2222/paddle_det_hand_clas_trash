import onnxruntime as ort
import cv2

import time

from utils.infer import PredictConfig, predict_image, classify_image
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--infer_det_cfg", type=str, default="./det/infer_cfg.yml", help="infer_cfg.yml")
    parser.add_argument(
        '--det_onnx_file', type=str, default="./det/hand_object_detect_picodet_s_320.onnx", help="onnx model file path")
    
    parser.add_argument("--infer_clas_cfg", type=str, default="./clas/infer_cfg.yml", help="infer_cfg.yml")
    parser.add_argument(
        '--clas_onnx_file', type=str, default="./clas/4clas_trash_MobileNetV3_x_1_0_95_7.onnx", help="onnx model file path")

    # Load the ONNX model
    FLAGS = parser.parse_args()

    # load detetction model
    predictor_det = ort.InferenceSession(FLAGS.det_onnx_file, providers=["CUDAExecutionProvider"])
    infer_config_det = PredictConfig(FLAGS.infer_det_cfg)


    # load classify model
    predictor_clas = ort.InferenceSession(FLAGS.clas_onnx_file)
    infer_config_clas = PredictConfig(FLAGS.infer_clas_cfg)


    # load camera
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        height, width = frame.shape[:2]
        print(height, width)
        t1 = time.time()
        boxes = predict_image(infer_config_det, predictor_det, frame, im_show=False)
        # print(boxes)
        
        for box in boxes:
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
            w0, h0 = int((x1-  x0)/2), int((y1 - y0)/2)
            
            y_min = (y0 - h0) if (y0 - h0) >0 else 0
            y_max = (y1 + h0) if (y1 + h0) < height else height
            x_min = (x0 - h0) if (x0 - h0) >0 else 0
            x_max = (x1 + h0) if (x1 + h0) < width else width

            img_crop = frame[y_min:y_max,x_min:x_max]
            clas, prob = classify_image(infer_config_clas, predictor_clas, frame)
            
            frame = cv2.putText(frame, str(clas) + " :" + str(prob), (x_min,y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255),1)
        
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
            
        print("FPS:" , 1/(time.time() - t1))