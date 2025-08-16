import onnx
from onnx import shape_inference, checker
from onnx import helper
import sys

src = '/workspace/tmp/qwen3_embedding/1/model.onnx'
dst = '/workspace/tmp/qwen3_embedding/1/model_inferred.onnx'

m = onnx.load(src)
# Ensure IR version is set
try:
    m.ir_version = onnx.IR_VERSION
except Exception:
    pass

# Run strict shape inference so TRT EP has concrete shapes for subgraphs
m = shape_inference.infer_shapes(m, strict_mode=True)

# Best-effort model check; proceed even if warnings
try:
    checker.check_model(m)
except Exception as e:
    print(f'CHECK WARNING: {e}', file=sys.stderr)

onnx.save(m, dst)
print('OK: wrote', dst)

