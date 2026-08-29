from __future__ import annotations

import os
import tempfile

import onnx


def optimize(model_path: str) -> str:
    try:
        import onnxruntime as ort
    except ImportError:
        return model_path

    out = os.path.join(tempfile.mkdtemp(prefix="onnx_opt_"), "optimized.onnx")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    opts.optimized_model_filepath = out
    # one thread: ORT otherwise logs an affinity failure per core
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    ort.set_default_logger_severity(3)
    try:
        ort.InferenceSession(model_path, opts,
                             providers=["CPUExecutionProvider"])
    except Exception:
        return model_path            # not fatal: the unfused graph still maps
    return out if os.path.exists(out) else model_path


def infer_shapes(model) -> dict[str, list[int]]:
    model = onnx.shape_inference.infer_shapes(model)
    shapes: dict[str, list[int]] = {}
    for init in model.graph.initializer:
        shapes[init.name] = list(init.dims)
    for group in (model.graph.input, model.graph.value_info, model.graph.output):
        for vi in group:
            dims = [d.dim_value if d.dim_value > 0 else 1
                    for d in vi.type.tensor_type.shape.dim]
            if dims:
                shapes[vi.name] = dims
    return shapes


def load(model_path: str, optimize_graph: bool = True):
    original = onnx.load(model_path)
    if not optimize_graph:
        return original, infer_shapes(original)

    path = optimize(model_path)
    if path == model_path:
        return original, infer_shapes(original)

    # ORT drops value_info, and shape inference cannot rebuild it because the
    # fused nodes it introduces are com.microsoft ops with no standard schema.
    # The unfused graph still names the tensors that survive, so its shapes
    # fill the gap; the optimized graph wins wherever both have an entry.
    model = onnx.load(path)
    shapes = {**infer_shapes(original), **infer_shapes(model)}
    return model, shapes
