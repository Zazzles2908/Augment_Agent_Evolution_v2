import sys
import onnx
from onnx import helper, TensorProto

"""
Patch an ONNX model to accept INT32 inputs for the given input tensor names.
It inserts Cast (to INT64) nodes immediately after each patched input and
rewires all downstream consumers to use the cast output. This keeps internal
expectations of INT64 (e.g., Gather indices) intact, while exposing INT32
bindings to TensorRT for engine build.

Usage:
  python patch_onnx_int32_inputs.py <model.onnx> [input_name1 input_name2 ...]
If no input names are provided, defaults to ["input_ids", "attention_mask"].
Writes a sibling file <model>.int32.onnx and leaves the original untouched.
"""

def main():
    if len(sys.argv) < 2:
        print("Usage: patch_onnx_int32_inputs.py <model.onnx> [input_names...]")
        sys.exit(2)

    path = sys.argv[1]
    inputs_to_patch = sys.argv[2:] or ["input_ids", "attention_mask"]

    m = onnx.load(path)
    # Ensure IR version is set to avoid checker complaints
    try:
        m.ir_version = onnx.IR_VERSION
    except Exception:
        pass

    # Build quick lookup for model inputs
    input_map = {i.name: i for i in m.graph.input}

    # Track name remaps: original_input_name -> cast_output_name
    remap = {}
    new_nodes = []

    for name in inputs_to_patch:
        inp = input_map.get(name)
        if inp is None:
            print(f"WARN: input '{name}' not found; skipping")
            continue

        # If not a tensor, skip
        tt = inp.type.tensor_type
        if tt.elem_type == TensorProto.INT32:
            print(f"INFO: input '{name}' already INT32; skipping")
            continue

        # Force external binding to INT32
        print(f"PATCH: setting input '{name}' elem_type to INT32 (was {tt.elem_type})")
        tt.elem_type = TensorProto.INT32

        # Insert Cast to INT64 for internal consumers
        cast_out = name + "__as_i64"
        cast_node = helper.make_node(
            "Cast",
            inputs=[name],
            outputs=[cast_out],
            name=f"Cast_{name}_to_int64",
            to=TensorProto.INT64,
        )
        new_nodes.append(cast_node)
        remap[name] = cast_out

    # Rewire all nodes' inputs if they consume patched inputs
    for node in m.graph.node:
        for idx, s in enumerate(node.input):
            if s in remap:
                node.input[idx] = remap[s]

    # Prepend new cast nodes by reconstructing nodes list (avoids slice assignment)
    nodes = list(m.graph.node)
    nodes = new_nodes + nodes
    m.graph.ClearField('node')
    m.graph.node.extend(nodes)

    out_path = path.replace('.onnx', '.int32.onnx')
    # Preserve external data layout to avoid breaking large initializers
    onnx.save_model(m, out_path, save_as_external_data=True)
    print(f"OK: wrote {out_path}")


if __name__ == "__main__":
    main()

