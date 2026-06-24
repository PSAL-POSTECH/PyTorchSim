"""lower_to_emitc pass (C4): skeleton+API MLIR -> EmitC -> C++ -> trace `.so`.

Second stage of the C++ trace pipeline (docs/design/togsim_cpp_trace.md, sec
5-7). Takes the skeleton+API module from `build_skeleton` (loop nest +
`togsim.*` ops) and produces an EmitC module whose single entry function

    extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n)

mirrors the loop skeleton, with every `togsim.*` op as an `emitc.call_opaque`
to the matching `togsim_runtime.h` free function (`togsim_ops.EMITC_CALLEE`).
`mlir-translate --mlir-to-cpp` renders it to C++, compiled to a `.so` that
exports `togsim_kernel` and leaves `togsim_dma/wait/compute/signal` undefined for
the TOGSim loader to resolve at `dlopen`.

How the lowering is done -- it drives the *upstream* EmitC conversion passes and
adds only the glue they cannot do:

  1. (python) Rewrite the unregistered `togsim.*` ops to `emitc.call_opaque`.
     Unregistered ops have no registered conversion patterns, so this must be a
     custom rewrite (design sec 8). Also rewrite the kernel's signature to the
     ABI form (drop the memref tensor args -- the trace producer never touches
     tensor data; base addresses are deferred to P3) and drop the aux
     globals / wrapper func.
  2. (upstream passes, in-process PassManager)
        func.func(lower-affine) -> convert-scf-to-emitc
        -> convert-arith-to-emitc -> convert-func-to-emitc
     This is the EmitC infrastructure: it lowers the affine/scf loop nest to
     `emitc.for`, the index/arith (loop bounds, and in P3 the address
     arithmetic) to EmitC, and the func to `emitc.func`.
  3. (python) Two small fixups the passes leave behind in this LLVM 20 build:
       * `convert-scf-to-emitc` emits `emitc.for` with `index`-typed bounds, so
         `convert-arith-to-emitc` (which makes constants `!emitc.size_t`) leaves
         `builtin.unrealized_conversion_cast` on the bounds that nothing folds
         and `mlir-to-cpp` cannot print (design sec 8 "EmitC coverage" risk).
         `_fold_for_bound_casts` rewrites those casts away.
       * add the `extern "C"` specifier so `dlsym` finds the entry unmangled.

Requires the MLIR Python bindings (incl. `mlir.passmanager`); the .cpp/.so
steps additionally require `mlir-translate` (TORCHSIM_LLVM_PATH) and a host C++
compiler.
"""

import os
import subprocess

from mlir.passmanager import PassManager

from . import togsim_ops as ts
from ._mlir_util import walk_ops, i32, i64, attr_int, attr_i64_array
from .build_tog import ir, _find_kernel

#: emitted entry symbol (== ts.ENTRY_SYMBOL == "togsim_kernel").
ENTRY = ts.ENTRY_SYMBOL

#: EmitC type of the opaque EmitCtx* threaded through every call.
CTX_TYPE = '!emitc.ptr<!emitc.opaque<"EmitCtx">>'

#: upstream EmitC conversion pipeline (the infrastructure this pass drives).
_PIPELINE = ("builtin.module("
             "func.func(lower-affine),"
             "convert-scf-to-emitc,"
             "convert-arith-to-emitc,"
             "convert-func-to-emitc)")

#: prepended to the mlir-to-cpp output; pulls in size_t/intN_t and the ABI.
_PRELUDE = (
    "#include <cstddef>\n"
    "#include <cstdint>\n"
    "using std::size_t;\n"
    '#include "togsim_runtime.h"\n'
)


# ---------------------------------------------------------------------------
# attribute builders / readers
# ---------------------------------------------------------------------------
def _idx(v):
    return ir.IntegerAttr.get(ir.IndexType.get(), int(v))


def _opaque(ctx, text):
    return ir.Attribute.parse('#emitc.opaque<"%s">' % text, ctx)


def _arr(ctx, vals):
    """A C compound-literal `(const int64_t[]){...}` arg, or `nullptr` if empty
    (the call site decays it to a `const int64_t*`)."""
    vals = list(vals)
    if not vals:
        return _opaque(ctx, "nullptr")
    return _opaque(ctx, "(const int64_t[]){%s}" % ", ".join(str(int(v)) for v in vals))


def _attr_bool(op, key):
    return 1 if ir.BoolAttr(op.operation.attributes[key]).value else 0


# ---------------------------------------------------------------------------
# step 1: rewrite signature + togsim.* ops (the unregistered-op glue)
# ---------------------------------------------------------------------------
def _strip_aux(module):
    """Erase memref.global decls and every func except @kernel (the wrapper)."""
    victims = []
    for op in module.body.operations:
        name = op.operation.name
        if name == "memref.global":
            victims.append(op)
        elif name == "func.func":
            if ir.StringAttr(op.operation.attributes["sym_name"]).value != "kernel":
                victims.append(op)
    for op in victims:
        op.operation.erase()


def _rewrite_signature(kernel, ctx):
    """Replace @kernel's memref tensor args with the ABI args
    (EmitCtx*, int64_t* shape_args, int32_t n) and rename it to togsim_kernel.
    Returns the ctx Value."""
    block = kernel.regions[0].blocks[0]
    for arg in block.arguments:
        if len(list(arg.uses)) > 0:
            raise ValueError(
                "kernel arg still used after build_skeleton; cannot drop it "
                "(expected the DCE to have removed all tensor-data ops)")
    # erase existing (memref) args high-to-low, then append the ABI args.
    for i in reversed(range(len(block.arguments))):
        block.erase_argument(i)
    ptr = ir.Type.parse(CTX_TYPE, ctx)
    i64ptr = ir.Type.parse("!emitc.ptr<i64>", ctx)
    i32 = ir.IntegerType.get_signless(32)
    loc = ir.Location.unknown(ctx)
    block.add_argument(ptr, loc)
    block.add_argument(i64ptr, loc)
    block.add_argument(i32, loc)
    kernel.operation.attributes["function_type"] = ir.TypeAttr.get(
        ir.FunctionType.get([ptr, i64ptr, i32], []))
    kernel.operation.attributes["sym_name"] = ir.StringAttr.get(ENTRY)
    return block.arguments[0]


def _call(ctx, ctx_val, op, callee, arg_attrs):
    """Insert emitc.call_opaque <callee>(ctx) {args=[0:index, ...]} before `op`.
    The leading `0 : index` references operand 0 (ctx); other entries are
    literal C args (integer attr -> literal, #emitc.opaque -> verbatim)."""
    ir.Operation.create(
        "emitc.call_opaque", results=[], operands=[ctx_val],
        attributes={"callee": ir.StringAttr.get(callee),
                    "args": ir.ArrayAttr.get([_idx(0)] + arg_attrs)},
        loc=ir.Location.unknown(ctx), ip=ir.InsertionPoint(op))


def _innermost_outer_loop(block):
    """Deepest `affine.for {outer_loop=true}` (the PARALLEL/ACCUMULATION
    boundary). Returns the op or None if the kernel has no parallel loop."""
    found = [None]

    def is_outer(op):
        a = op.operation.attributes
        return "outer_loop" in a and ir.BoolAttr(a["outer_loop"]).value

    def walk(b):
        for op in b.operations:
            if op.operation.name == "affine.for" and is_outer(op):
                found[0] = op   # nested outer loops overwrite -> deepest wins
            for r in op.operation.regions:
                for bb in r.blocks:
                    walk(bb)

    walk(block)
    return found[0]


def _insert_core_alloc(ctx, kernel, ctx_val):
    """Insert `togsim_core_alloc(ctx)` at the start of each parallel work-item:
    the first op of the innermost PARALLEL loop body (or the kernel entry if the
    kernel has no parallel loop -> a single work-item). The runtime binds the
    following ops to the returned core (sec 9.3); the producer never names
    num_cores. The return value is discarded (no free -- a core is an assignment,
    not a held resource)."""
    block = kernel.regions[0].blocks[0]
    target = _innermost_outer_loop(block)
    body = target.operation.regions[0].blocks[0] if target is not None else block
    ir.Operation.create(
        "emitc.call_opaque", results=[], operands=[ctx_val],
        attributes={"callee": ir.StringAttr.get(ts.CORE_ALLOC_CALLEE),
                    "args": ir.ArrayAttr.get([_idx(0)])},
        loc=ir.Location.unknown(ctx),
        ip=ir.InsertionPoint.at_block_begin(body))


def _rewrite_togsim_ops(ctx, kernel, ctx_val):
    block = kernel.regions[0].blocks[0]
    victims = []
    for op in walk_ops(block):
        name = op.operation.name
        ipo = ir.InsertionPoint(op)
        if name == ts.DMA:
            dims = attr_i64_array(op, ts.ATTR_DIMS)
            # The DRAM element offset is the togsim.dma operand (the original
            # affine index, kept live by build_skeleton); pass it as a call
            # operand so convert-arith-to-emitc lowers the address arithmetic
            # into the producer (P3 approach A). The runtime adds the tensor base.
            # Operands carried by build_skeleton: [dram_index, tag_index] (each
            # optional). Pass each as a call operand so convert-arith-to-emitc
            # lowers it; reference it from `args` by its operand position. offset
            # -> DRAM byte address (runtime adds the tensor base); tag_slot -> the
            # SRAM tile slot (runtime uses it for double-buffer/SRAM-capacity).
            ins = list(op.operation.operands)
            dram_operand = ins[0] if len(ins) >= 1 else None
            tag_operand = ins[1] if len(ins) >= 2 else None
            operands = [ctx_val]
            offset_arg = i64(0)
            tag_arg = i64(0)
            if dram_operand is not None:
                operands.append(dram_operand)
                offset_arg = _idx(len(operands) - 1)
            if tag_operand is not None:
                operands.append(tag_operand)
                tag_arg = _idx(len(operands) - 1)
            args = [_idx(0),
                    i32(attr_int(op, ts.ATTR_DIR)),
                    i32(attr_int(op, ts.ATTR_ARG_ID)),
                    offset_arg,
                    i32(len(dims)),
                    _arr(ctx, dims),
                    _arr(ctx, attr_i64_array(op, ts.ATTR_STRIDES)),
                    i32(attr_int(op, ts.ATTR_ELEM_BITS)),
                    i32(_attr_bool(op, ts.ATTR_IS_ASYNC)),
                    i32(attr_int(op, ts.ATTR_TAG_ID)),
                    tag_arg]
            _rb = attr_i64_array(op, ts.ATTR_READ_BUFS)
            _wb = attr_i64_array(op, ts.ATTR_WRITE_BUFS)
            args += [_arr(ctx, _rb), i32(len(_rb)), _arr(ctx, _wb), i32(len(_wb))]
            # togsim_dma is void: the dma is paired with its barrier by the runtime
            # (tag_id, tag_slot), not a returned handle.
            ir.Operation.create(
                "emitc.call_opaque", results=[], operands=operands,
                attributes={"callee": ir.StringAttr.get(ts.EMITC_CALLEE[ts.DMA]),
                            "args": ir.ArrayAttr.get(args)},
                loc=ir.Location.unknown(ctx), ip=ipo)
            victims.append(op)
        elif name == ts.MEMORY_BAR:
            # explicit async-DMA sync (the original dma_wait) ->
            # togsim_memory_barrier(ctx, tag_id, tag_slot, write_bufs). The tag
            # index operand (if any) is the runtime tag slot.
            ins = list(op.operation.operands)
            operands = [ctx_val]
            tag_arg = i64(0)
            if ins:
                operands.append(ins[0])
                tag_arg = _idx(len(operands) - 1)
            _wb = attr_i64_array(op, ts.ATTR_WRITE_BUFS)
            ir.Operation.create(
                "emitc.call_opaque", results=[], operands=operands,
                attributes={"callee": ir.StringAttr.get(ts.EMITC_CALLEE[ts.MEMORY_BAR]),
                            "args": ir.ArrayAttr.get(
                                [_idx(0), i32(attr_int(op, ts.ATTR_TAG_ID)), tag_arg,
                                 _arr(ctx, _wb), i32(len(_wb))])},
                loc=ir.Location.unknown(ctx), ip=ipo)
            victims.append(op)
        elif name == ts.COMPUTE:
            # skeleton compute carries no dims (cost is keyed by tile_id) -> 0/null.
            _rb = attr_i64_array(op, ts.ATTR_READ_BUFS)
            _wb = attr_i64_array(op, ts.ATTR_WRITE_BUFS)
            _call(ctx, ctx_val, op, ts.EMITC_CALLEE[ts.COMPUTE],
                  [i64(attr_int(op, ts.ATTR_TILE_ID)),
                   i32(attr_int(op, ts.ATTR_COMPUTE_TYPE)),
                   i32(0), _opaque(ctx, "nullptr"),
                   _arr(ctx, _rb), i32(len(_rb)), _arr(ctx, _wb), i32(len(_wb))])
            victims.append(op)
        elif name == ts.COMPUTE_BAR:
            # explicit compute fence -> togsim_compute_barrier(ctx) (sec 10.7).
            ir.Operation.create(
                "emitc.call_opaque", results=[], operands=[ctx_val],
                attributes={"callee": ir.StringAttr.get(ts.EMITC_CALLEE[ts.COMPUTE_BAR]),
                            "args": ir.ArrayAttr.get([_idx(0)])},
                loc=ir.Location.unknown(ctx), ip=ipo)
            victims.append(op)
    for op in victims:
        op.operation.erase()


# ---------------------------------------------------------------------------
# step 3: post-conversion fixups
# ---------------------------------------------------------------------------
def _retype_for_to_size_t(module):
    """Make every `emitc.for` use `!emitc.size_t` bounds + induction variable,
    then drop the `index`<->`!emitc.size_t` `unrealized_conversion_cast` ops that
    `convert-scf-to-emitc` / `convert-arith-to-emitc` leave behind (mlir-to-cpp
    cannot print them; --reconcile cannot fold them).

    `emitc.for` accepts `size_t` bounds with the explicit type, and a `size_t` IV
    makes the lowered address arithmetic (`convert-arith-to-emitc`, which works
    in `size_t`) cast-free. So: set each IV to size_t, then for every
    index<->size_t cast replace its result with its source (every consumer here
    -- `emitc.for` bounds, `emitc.call_opaque` operands, `emitc` arith -- accepts
    either, and after the IV retype each such cast bridges equal types)."""
    idx = ir.IndexType.get()
    st = ir.Type.parse("!emitc.size_t", module.context)

    for op in list(walk_ops(module.body)):
        if op.operation.name == "emitc.for":
            op.operation.regions[0].blocks[0].arguments[0].set_type(st)

    dead = []
    for op in list(walk_ops(module.body)):
        if op.operation.name != "builtin.unrealized_conversion_cast":
            continue
        res = op.results[0]
        src = list(op.operation.operands)[0]
        # idx<->size_t bridges (incl. the size_t->size_t identities left after
        # the IV retype): every consumer here accepts either, so fold to source.
        if src.type in (idx, st) and res.type in (idx, st):
            res.replace_all_uses_with(src)
            dead.append(op)
    for d in dead:
        try:
            d.operation.erase()
        except Exception:
            pass


def _add_extern_c(module, ctx):
    for op in module.body.operations:
        if (op.operation.name == "emitc.func"
                and ir.StringAttr(op.operation.attributes["sym_name"]).value == ENTRY):
            op.operation.attributes["specifiers"] = ir.ArrayAttr.get(
                [ir.StringAttr.get('extern "C"')])
            return
    raise ValueError("emitc.func @%s not found after conversion" % ENTRY)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def lower_to_emitc(skeleton_module):
    """Lower a skeleton+API module (in place) to an EmitC module with the
    `togsim_kernel` entry function. Returns the same module."""
    ctx = skeleton_module.context
    kernel = _find_kernel(skeleton_module)
    if kernel is None:
        raise ValueError("no @kernel found in skeleton module")

    _strip_aux(skeleton_module)
    ctx_val = _rewrite_signature(kernel, ctx)
    _insert_core_alloc(ctx, kernel, ctx_val)          # core_alloc per work-item
    _rewrite_togsim_ops(ctx, kernel, ctx_val)

    PassManager.parse(_PIPELINE, ctx).run(skeleton_module.operation)

    _retype_for_to_size_t(skeleton_module)
    _add_extern_c(skeleton_module, ctx)
    return skeleton_module


# ---------------------------------------------------------------------------
# C++ / .so backend
# ---------------------------------------------------------------------------
def _mlir_translate_bin():
    return os.path.join(os.environ.get("TORCHSIM_LLVM_PATH", "/usr/bin"),
                        "mlir-translate")


def emitc_to_cpp(emitc_module, mlir_translate=None):
    """Render `emitc_module` to C++ source (prelude + mlir-to-cpp body)."""
    mlir_translate = mlir_translate or _mlir_translate_bin()
    proc = subprocess.run(
        [mlir_translate, "--mlir-to-cpp"],
        input=str(emitc_module), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("mlir-translate --mlir-to-cpp failed:\n" + proc.stderr)
    return _PRELUDE + proc.stdout


def compile_so(cpp_text, so_path, include_dir, cxx=None):
    """Compile producer C++ to `so_path`. `include_dir` must hold
    togsim_runtime.h. togsim_* symbols are left undefined (resolved at dlopen)."""
    cxx = cxx or os.environ.get("CXX", "g++")
    cpp_path = os.path.splitext(so_path)[0] + ".cpp"
    with open(cpp_path, "w") as fh:
        fh.write(cpp_text)
    proc = subprocess.run(
        [cxx, "-shared", "-fPIC", "-std=gnu++17", "-O2",
         "-I", include_dir, cpp_path, "-o", so_path],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("%s failed:\n%s" % (cxx, proc.stderr))
    return so_path


def _default_include_dir():
    root = os.environ.get("TORCHSIM_DIR")
    if not root:
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(root, "TOGSim", "include")


def skeleton_to_so(skeleton_module, so_path, include_dir=None):
    """skeleton module -> EmitC -> C++ -> compiled trace `.so`. Returns the
    EmitC module text (for inspection / caching)."""
    emitc = lower_to_emitc(skeleton_module)
    cpp = emitc_to_cpp(emitc)
    compile_so(cpp, so_path, include_dir or _default_include_dir())
    return str(emitc)


def build_trace_so(postvcix_path, so_path, include_dir=None):
    """Full P2 path from a post-vcix kernel .mlir to a trace `.so`."""
    from . import build_skeleton as bs

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(postvcix_path).read(), ctx)
        bs.build_skeleton(module)
        return skeleton_to_so(module, so_path, include_dir)


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(prog="lower_to_emitc.py")
    parser.add_argument("input", help="post-vcix kernel .mlir")
    parser.add_argument("--so", required=True, help="output .so path")
    parser.add_argument("--include-dir", default=None,
                        help="dir holding togsim_runtime.h (default: TOGSim/include)")
    parser.add_argument("--emit-cpp", default=None,
                        help="also write the generated C++ here")
    parser.add_argument("--emit-mlir", default=None,
                        help="also write the EmitC MLIR here")
    args = parser.parse_args(argv[1:])

    from . import build_skeleton as bs
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(args.input).read(), ctx)
        bs.build_skeleton(module)
        emitc = lower_to_emitc(module)
        if args.emit_mlir:
            open(args.emit_mlir, "w").write(str(emitc))
        cpp = emitc_to_cpp(emitc)
        if args.emit_cpp:
            open(args.emit_cpp, "w").write(cpp)
        compile_so(cpp, args.so, args.include_dir or _default_include_dir())
    import sys
    sys.stderr.write("wrote %s\n" % args.so)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
