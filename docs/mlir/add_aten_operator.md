# How to add support for a new ATen Operator in PyTorchSim

---

## Overview
PyTorchSim executes PyTorch programs by **lowering ATen operators into MLIR**,
followed by backend-specific code generation and simulation.
This wiki describes contributors through the process of adding support for **new
ATen operators** by defining custom lowerings in PyTorchSim’s MLIR-based
execution path.


We use a dummy operator, `torch._foobar()`, as a minimal example.
Although `_foobar` has trivial semantics, it still exercises the full
integration workflow:

- Defining an MLIR template  
- Adding a custom lowering  
- Registering the lowering  
- Validating correctness with a test  


---
## Background

Before diving into the step-by-step implementation, it helps to understand
how an ATen operator flows through PyTorchSim’s MLIR-based pipeline. At a high level, supporting a new ATen operator means intercepting the
operator during the lowering stage and redirecting it to a custom MLIR template.
Understanding this flow makes the implementation steps clearer.

At a high level, adding support for a new ATen op means intercepting this flow
at the lowering stage and redirecting it to a custom MLIR template.

### Graph capture (`test_<op>.py`)

When `torch.compile` captures a Python function, each `torch.<op>(...)` call
is recorded in the graph as an ATen operator (`aten.<op>`).

This is the first point where the operator becomes visible to the lowering
pipeline and eligible for custom handling.

### Lowering stage (`mlir_lowering.py`)

During lowering, a custom function (e.g., `custom_<op>`) is invoked for
`aten.<op>`.

The lowering:
- materializes inputs in Inductor IR if necessary,
- constructs an MLIR template instance from `mlir_<op>_template.py`,
- replaces the original ATen op with a template-backed Inductor IR node.

This is the key hook where new ATen operator support is introduced.

### Scheduling stage (`mlir_scheduling.py`)

The template-backed node enters the scheduler and is routed through
`codegen_template()`.

At this stage:
- scheduling decisions are applied,
- the MLIR source code for the kernel is generated,
- the kernel is prepared for registration.

### Kernel registration (`define_kernel()`)

The generated MLIR source (`src_code`) is registered as a compilable and
cacheable kernel via the wrapper.

After registration, the kernel becomes part of the code generation artifacts
and can be reused across runs without regenerating MLIR.



---

## Table of Contents
 
- [Toy Example: `_foobar` as a Dummy ATen Op](#toy-example-_foobar-as-a-dummy-aten-op)
- [Step 1 — Create `mlir_<op>_template.py`](#step-1--create-mlir_op_templatepy)
- [Step 2 — Add a Custom Lowering in `mlir_lowering.py`](#step-2--add-a-custom-lowering-in-mlir_loweringpy)
  - [Step 2.1 — Define the Custom Lowering Function](#step-21--define-the-custom-lowering-function)
  - [Step 2.2 — Register the Lowering via `loweringsupdate`](#step-22--register-the-lowering-via-loweringsupdate)
- [Step 3 — Add a Test (`test_<op>.py`)](#step-3--add-a-test-test_oppy)
- [Summary](#summary)
<!-- - [Verifying the MLIR Output](#verifying-the-mlir-output) -->
<!-- - [Extending to Real-World (Complex) ATen Ops](#extending-to-real-world-complex-aten-ops) -->
<!-- - [Common Pitfalls and Edge Cases](#common-pitfalls-and-edge-cases) -->
<!-- - [Checklist: When a Custom Lowering Is Necessary](#checklist-when-a-custom-lowering-is-necessary) -->





---
## Toy Example: `_foobar` as a Dummy ATen Op

`_foobar` is a deliberately trivial ATen operator exposed in PyTorch as `torch._foobar`
(and also accessible via `torch.ops.aten._foobar`).

### Signature

From the PyTorch C++ API, `_foobar` has the following signature:

* `at::_foobar(const Tensor& self, bool arg1=true, bool arg2=true)`

In other words, it takes:
- one input tensor (`self`)
- three optional boolean flags (`arg1`, `arg2`)

### Behavior

In our Python checks, `_foobar` behaves like a simple **copy/identity-style op**:
- the output has the same shape and dtype as the input
- the output values match the input values


Once you understand the full flow with `_foobar`, you can replace it with other ATen op and expand
the lowering/template logic as needed.

---



## Step 1 — Create `mlir_<op>_template.py`

This step defines an MLIR template for the new ATen operator.  
For `_foobar`, the template implements a minimal **identity-style kernel** that copies input elements to output elements.

We walk through the file top-down, highlighting the role of each section.


### File skeleton and imports  

Defines the core dependencies used to build an MLIR template, including:
- the base `MLIRTemplate` / `MLIRTemplateKernel` classes
- shared utilities from `mlir_common`
- symbolic helpers (e.g., `sympy`) for shape expressions

```python
from typing import List, Optional
import sympy

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate, MLIRTemplateKernel
from torch._inductor.ir import IRNode, Buffer
from PyTorchSimFrontend.mlir import mlir_common
```

### MLIR string template

Defines the raw MLIR code as a string template.
This template contains symbolic placeholders such as:

* number of elements in X (`{{ M }}`)

* tile size (`{{ TILE }}`)

* input/output memref shapes

In this foobar example, the kernel performs 1D tiling over M and, within each tile, copies elements one by one from X to Y. It does not use SRAM buffers and does not emit DMA ops (MVIN/MVOUT) — all accesses are direct DRAM `memref.load`/`memref.store` operations. The placeholders are filled later via `kernel.render_options`. For more complex ATen ops (e.g., multi‑dimensional tiling, SRAM/DMA usage, prologue/epilogue fusion), see the next WIKI page.


```python
TEMPLATE = r"""
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=[X], outputs=[Y], names_str="X, Y", input_reorder=input_reorder)}} {
  %M_const = arith.constant {{ M }} : index
  affine.for %index0 = 0 to {{ M }} step {{ TILE }} {
    affine.for %t = 0 to {{ TILE }} step 1 {
      %g = arith.addi %index0, %t : index
      %cond = arith.cmpi slt, %g, %M_const : index
      scf.if %cond {
        %val = memref.load %X[%g] : {{ X_flat_mlir_shape }}
        memref.store %val, %Y[%g] : {{ Y_flat_mlir_shape }}
      }
    }
  } { outer_loop=true }
  return
}
"""
```



### Template class definition & `__init__`

Defines the `MLIRFoobarTemplate` class, which inherits from `MLIRTemplate`.
Initializes the template by calling the base class constructor to register.

```python
class MLIRFoobarTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, input_reorder=None):
        # Initialize the MLIR template with the kernel name and input/output nodes.
        super().__init__("kernel", input_nodes, layout, input_reorder)
```

### Render entry point (`render`)

Defines the main render entry point for the template. It selects the output node (template buffer/epilogue), prepares tile descriptors and indices, fills `kernel.render_options`, and returns the rendered MLIR code.

```python
    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               prologue_nodes: Optional[List[IRNode]] = None,
               tile_info = None,
               **kwargs):
```

### Output selection and basic setup

Selects the output buffer, binds symbolic names for input (`X`) and output (`Y`),
computes the number of elements in X (`M`), and derives the tile size (`TILE`) from the kernel
configuration.

```python
        if template_buffer_node is not None:
            self.output_node = template_buffer_node  
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = epilogue_nodes[-1]  

        X = self.input_nodes[0]  
        Y = self.output_node 

        M = X.get_numel()  
        TILE = kernel.vector_lane  
```

### Tile descriptors and indices

Defines tile descriptors for the input and output tensors.
A tile descriptor (`MLIRMultiDimTile`) is a small metadata object that captures a tile’s shape, stride, and vector‑lane mapping. It is later used to form SRAM buffer types, DMA parameters, and indexing decisions. In this _foobar example, both X and Y use 1D tiles of size TILE, and both are indexed by the same loop variable (`index0`) to represent elementwise copy.


```python
        vlane_stride = 1
        vlane_split_axis = 0
        X_tile_size = [TILE]
        X_tile_stride = [1]
        X_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        X_tile_desc.set_tile_size_stride(X_tile_size, X_tile_stride)
        X_tile_desc.set_name("X_buffer")
        X_idx = [sympy.Symbol("index0")]

      
        Y_tile_size = [TILE]
        Y_tile_stride = [1]
        Y_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Y_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
        Y_tile_desc.set_name("Y_buffer")
        Y_idx = [sympy.Symbol("index0")]
```

### Memref shape strings

Defines the memref type strings used in the MLIR template for both input and output.
A memref is MLIR’s memory reference type, it describes a buffer in memory by its shape and element type and optionally layout/stride information.
For example, memref<128xf32> is a 1D buffer of 128 floats. In this foobar example, both X and Y are treated as 1D buffers, so we use memref<{M}xf32>:

```python
        X_flat_mlir_shape = f"memref<{M}x{{DATA_STYPE}}>".replace('{DATA_STYPE}', 'f32')
        Y_flat_mlir_shape = f"memref<{M}x{{DATA_STYPE}}>".replace('{DATA_STYPE}', 'f32')
```

### Render options

Collects all symbolic values and configuration parameters into
`kernel.render_options`. These options are later used to render the MLIR string
template.

```python
        kernel.render_options = dict(
            KERNEL_NAME=self.name,  
            kernel=kernel,  
            M=M,  
            TILE=TILE,  
            X=X, 
            Y=Y,  
            X_idx=X_idx,  
            Y_idx=Y_idx,  
            X_tile_desc=X_tile_desc, 
            Y_tile_desc=Y_tile_desc,  
            X_flat_mlir_shape=X_flat_mlir_shape,  
            Y_flat_mlir_shape=Y_flat_mlir_shape,  
            DATA_STYPE="f32", 
            input_reorder=self.input_reorder,  
        )
```

### Epilogue 

Records metadata related to output buffers and element counts, which is useful for
exception handling and debugging.

```python
        kernel.epilogue_info = dict(
            output_node=self.output_node.name, 
            sram_var="Y_buffer",  
            dram_var="Y",  
            dram_tile_desc=Y_tile_desc, 
        )

```

### Render MLIR and add loop metadata & Return

Renders the final MLIR code by substituting placeholders in the template. 
And returns the final MLIR string that will be consumed by the kernel.

```python
        code = self._template_from_string(TEMPLATE).render(**kernel.render_options)

        return code
```

### Full `mlir_foobar_template.py`

Copy-paste the full reference implementation below to create `mlir_foobar_template.py`.

```python
from typing import List, Optional
import sympy

from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate, MLIRTemplateKernel
from torch._inductor.ir import IRNode, Buffer
from PyTorchSimFrontend.mlir import mlir_common


TEMPLATE = r"""
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=[X], outputs=[Y], names_str="X, Y", input_reorder=input_reorder)}} {
  %M_const = arith.constant {{ M }} : index
  affine.for %index0 = 0 to {{ M }} step {{ TILE }} {
    affine.for %t = 0 to {{ TILE }} step 1 {
      %g = arith.addi %index0, %t : index
      %cond = arith.cmpi slt, %g, %M_const : index
      scf.if %cond {
        %val = memref.load %X[%g] : {{ X_flat_mlir_shape }}
        memref.store %val, %Y[%g] : {{ Y_flat_mlir_shape }}
      }
    }
  } { outer_loop=true }
  return
}
"""

class MLIRFoobarTemplate(MLIRTemplate):

    def __init__(self, input_nodes, layout, input_reorder=None):
        # Initialize the MLIR template with the kernel name and input/output nodes.
        super().__init__("kernel", input_nodes, layout, input_reorder)

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               prologue_nodes: Optional[List[IRNode]] = None,
               tile_info = None,
               **kwargs):
        """Render the MLIR code for the `torch._foobar()` operation.

        This method generates the MLIR code by filling in the placeholders in the
        `TEMPLATE` string with the appropriate values for the input/output tensors,
        tile sizes, and other parameters.
        """
        if template_buffer_node is not None:
            self.output_node = template_buffer_node  
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = epilogue_nodes[-1]  

        X = self.input_nodes[0] 
        Y = self.output_node  

        M = X.get_numel() 
        TILE = kernel.vector_lane 

        vlane_stride = 1
        vlane_split_axis = 0
        X_tile_size = [TILE]
        X_tile_stride = [1]
        X_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        X_tile_desc.set_tile_size_stride(X_tile_size, X_tile_stride)
        X_tile_desc.set_name("X_buffer")
        X_idx = [sympy.Symbol("index0")]

        Y_tile_size = [TILE]
        Y_tile_stride = [1]
        Y_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Y_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
        Y_tile_desc.set_name("Y_buffer")
        Y_idx = [sympy.Symbol("index0")]

        X_flat_mlir_shape = f"memref<{M}x{{DATA_STYPE}}>".replace('{DATA_STYPE}', 'f32')
        Y_flat_mlir_shape = f"memref<{M}x{{DATA_STYPE}}>".replace('{DATA_STYPE}', 'f32')

        kernel.render_options = dict(
            KERNEL_NAME=self.name,  
            kernel=kernel,  
            M=M,  
            TILE=TILE,  
            X=X, 
            Y=Y,  
            X_idx=X_idx,  
            Y_idx=Y_idx,  
            X_tile_desc=X_tile_desc, 
            Y_tile_desc=Y_tile_desc,  
            X_flat_mlir_shape=X_flat_mlir_shape,  
            Y_flat_mlir_shape=Y_flat_mlir_shape,  
            DATA_STYPE="f32", 
            input_reorder=self.input_reorder,  
        )

        kernel.epilogue_info = dict(
            output_node=self.output_node.name, 
            sram_var="Y_buffer",  
            dram_var="Y",  
            dram_tile_desc=Y_tile_desc, 
        )

        code = self._template_from_string(TEMPLATE).render(**kernel.render_options)

        return code 
```


---

## Step 2 — Add a Custom Lowering in `mlir_lowering.py`

This step connects the ATen operator (`aten._foobar`) to the MLIR template.
All changes in this step are made in **`mlir_lowering.py`**.

It consists of two parts:
1. Defining the custom lowering function, and
2. Registering that function so Inductor actually uses it.

---

### Step 2.1 — Define the Custom Lowering Function

The custom lowering function specifies **how `aten._foobar` should be lowered**
during the Inductor lowering stage.

For this tutorial, the lowering is intentionally minimal and preserves the
MLIR/template path, making it suitable for testing and demonstration purposes.

```python
def custom_foobar(a, *args, **kwargs):
    a.realize()
    layout = a.layout
    mlir_template = MLIRFoobarTemplate([a], layout)
    return mlir_template.generate().output_node()
```

This function follows the standard pattern for custom lowerings:

* `a` (first argument):
The actual input tensor. During lowering, this is typically wrapped in a
`TensorBox`, which is Inductor’s IR wrapper that carries the tensor along
with its layout/metadata and deferred computation context.

* `a.realize()`:
Materializes the input so the MLIR template sees a concrete buffer / IR node.
This is a safe default pattern to ensure shape/stride/layout metadata is available.

* `layout = a.layout`:
Gets the Inductor layout object, which encapsulates device, dtype,
shape (size), and stride information. The MLIR template uses this to build
memref types and derive tiling/indexing behavior.

* `MLIRFoobarTemplate([a], layout)`:
Instantiates the MLIR template with the input node and its layout.

`generate().output_node()`:
Builds the template-backed Inductor IR node and returns it as the lowering
result.

This pattern is the baseline shape you will reuse for other operators, adding
more logic as the operator becomes more complex.

---

### Step 2.2 — Register the Lowering via `lowerings.update(...)`

Defining the lowering function alone is not enough.
It must be registered so that Inductor actually invokes it.

```python
lowerings.update({getattr(aten._foobar, overload): custom_foobar for overload in aten._foobar.overloads()})
```
The `lowerings` table is consulted by Inductor during the lowering phase.
When the ATen graph contains `aten._foobar`, Inductor looks up the operator in
this table and invokes `custom_foobar` instead of the default lowering.

Using `aten._foobar.overloads()` ensures that all overload variants of the
operator are covered, even if multiple signatures exist.

This is the wiring step that activates the MLIR template path defined in
Step 1 and implemented in Step 2.1.

---
## Step 3 — Add a Test (`test_<op>.py`)

This step validates the custom lowering end-to-end.
The test ensures that the operator is correctly captured, lowered, compiled,
and executed through the PyTorchSim MLIR path.

### Test helper for correctness

Defines a small helper function that compares the compiled output against a
CPU reference result and reports pass/fail status.
This helper can be reused across multiple operator tests.

```python
import torch
import torch._dynamo

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)
```

### Define a small wrapper function

Defines a minimal wrapper function (vector_foobar) that calls
torch._foobar(a).

This function is what torch.compile captures into the graph, making it the
entry point for Dynamo → ATen → Inductor → custom lowering.

```
def test_foobar(device, size=(128, 128)):
    def vector_foobar(a):
        return torch._foobar(a)
```

### Create input and compile

Creates a random input tensor and compiles the wrapper function using
torch.compile(dynamic=False).

This ensures the execution path goes through:
Dynamo → ATen → Inductor → custom lowering → MLIR template.

```python
    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vector_foobar)
    res = opt_fn(x)
```

### Run and verify

Executes the compiled function and compares the result against a reference
output.

For _foobar, the reference output is simply the input tensor moved to CPU,
since the operator behaves like an identity op.

```python
    out = x.cpu()
    test_result("Foobar", res, out)
```

### `__main__`

Defines the command-line entry point for the test.
This section:

* Sets up the custom PyTorchSim device and runner,

* Parses shape arguments,

* Runs the test across multiple input sizes.

```python
if __name__ == "__main__":
    import os
    import sys
    import argparse
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    parser = argparse.ArgumentParser(description="Run Foobar test with dynamic shape")
    parser.add_argument('--shape', type=str, default="(512,768)")
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.strip('()').split(',')))

    from Scheduler.scheduler import PyTorchSimRunner
    module = PyTorchSimRunner.setup_device()
    device = module.custom_device()
    test_foobar(device, (1, 1))
    test_foobar(device, (47, 10))
    test_foobar(device, (128, 128))
    test_foobar(device, shape)
```

Full `test_foobar.py`
Copy-paste the full reference implementation below to create `test_foobar.py`.

```python
import torch
import torch._dynamo

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

def test_foobar(device, size=(128, 128)):
    def vector_foobar(a):
        return torch._foobar(a)

    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vector_foobar)
    res = opt_fn(x)

    out = x.cpu()
    test_result("Foobar", res, out)


if __name__ == "__main__":
    import os
    import sys
    import argparse
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    parser = argparse.ArgumentParser(description="Run Foobar test with dynamic shape")
    parser.add_argument('--shape', type=str, default="(512,768)")
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.strip('()').split(',')))

    from Scheduler.scheduler import PyTorchSimRunner
    module = PyTorchSimRunner.setup_device()
    device = module.custom_device()
    test_foobar(device, (1, 1))
    test_foobar(device, (47, 10))
    test_foobar(device, (128, 128))
    test_foobar(device, shape)
```
---
## Summary

To add support for a new ATen operator in PyTorchSim’s MLIR path, you:

1. Define an MLIR template (`mlir_<op>_template.py`),
2. Implement and register a custom lowering in `mlir_lowering.py`,
3. Validate the integration with a dedicated test (`test_<op>.py`).

During testing, `torch.compile` captures the operator into the ATen graph.
The custom lowering replaces the ATen op with a template-backed Inductor IR node,
which then flows through scheduling, MLIR code generation, and kernel
registration before execution.

The `_foobar` example illustrates the complete integration flow for adding a new
ATen operator in PyTorchSim. You can use this example as a reference when
extending PyTorchSim to support additional ATen operators. For operators with
more complex semantics, refer to the follow-up documentation for guidance on
advanced lowering, layout handling, and performance considerations.

