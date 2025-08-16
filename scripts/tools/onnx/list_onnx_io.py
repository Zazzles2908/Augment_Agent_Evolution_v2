import onnx
import sys
p = sys.argv[1]
m = onnx.load(p)
print('IR_VERSION:', getattr(m, 'ir_version', None))
print('OPSET_IMPORT:', [(imp.domain, imp.version) for imp in m.opset_import])
print('INPUTS:')
for i in m.graph.input:
    shape = [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
    print(i.name, shape)
print('OUTPUTS:')
for o in m.graph.output:
    shape = [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]
    print(o.name, shape)

