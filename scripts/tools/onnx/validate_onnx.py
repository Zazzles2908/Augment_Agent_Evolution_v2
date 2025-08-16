import sys
import onnx
from onnx import shape_inference

if len(sys.argv) < 2:
    print("Usage: validate_onnx.py <path_to_onnx>")
    sys.exit(2)

p = sys.argv[1]
m = onnx.load(p)

# Basic model check
onnx.checker.check_model(m)
print("onnx.checker: OK")

# Try strict shape inference first, then fallback to non-strict
try:
    m2 = shape_inference.infer_shapes(m, strict_mode=True)
    print("shape_inference (strict): OK")
except Exception as e:
    print("shape_inference (strict): FAIL", e)
    try:
        m2 = shape_inference.infer_shapes(m)
        print("shape_inference (non-strict): OK")
    except Exception as e2:
        print("shape_inference (non-strict): FAIL", e2)

