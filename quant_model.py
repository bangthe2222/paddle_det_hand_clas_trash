import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load(r'D:\MAIN_DOCUMENTS\HCMUT K21\EduBin\MAIN\MODEL\PPCLAS_4\clas\4clas_trash_pplcnet_x1_0.onnx')

# convert model
model_simp, check = simplify(model)
onnx.save_model(model_simp,"model_det_sim.onnx")
assert check, "Simplified ONNX model could not be validated"