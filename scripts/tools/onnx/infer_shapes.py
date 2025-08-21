import onnx
from onnx import shape_inference, checker
m = onnx.load('/workspace/tmp/qwen3_embedding/1/model.onnx')
m = shape_inference.infer_shapes(m)
checker.check_model(m)
onnx.save(m, '/workspace/tmp/qwen3_embedding/1/model_inferred.onnx')
print('OK')
