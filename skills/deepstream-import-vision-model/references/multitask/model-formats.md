<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Model formats and conversion routes

The final DeepStream inference artifact is a TensorRT engine loaded by
`nvinfer`, or an ONNX model that `nvinfer` can build at runtime. Framework
files are source artifacts, not proof of DeepStream compatibility.

## Decision order

1. Use a supplied `.engine`/`.plan` only when its TensorRT version, GPU
   compatibility, input profile, precision, and plugin dependencies are known.
2. Otherwise use a supplied `.onnx` and run checker, shape inference, operator
   inspection, and a TensorRT build.
3. For PyTorch/SafeTensors/TorchScript, prefer the framework's supported ONNX
   exporter with explicit task metadata and fixed example inputs.
4. For TensorFlow/Keras/TFLite, use `tf2onnx` only as an experimental route;
   test every output and unsupported operator before building an engine.
5. Never silently convert an unknown or legacy format. Ask for a wrapper,
   sample input, output schema, and task definition.

## Accepted formats

### TensorRT engine or plan

Validate with the installed `trtexec` and capture the TensorRT version. Check
that the engine was built for the target GPU or is a compatible portable plan.
An engine can be used directly only when its source provenance is recorded and
the input tensor name, shape/profile, dtype, and output bindings match the
manifest. The caller must explicitly approve a canonical local model root.
Require the engine to be a regular file contained under that root without any
symbolic-link component before deserialization. Do not rebuild or overwrite a
user-provided engine, and do not run ONNX tooling against this direct branch.

### ONNX

Only the standard `.onnx` suffix is accepted by the unified visual-model route.

Run `onnx.checker.check_model`, `onnx.shape_inference.infer_shapes`, inspect
the graph inputs/outputs and opset, and build with `trtexec`. Keep dynamic
batch and spatial profiles explicit. A symbolic dimension is not a license to
guess: provide numeric `min_shape`, `opt_shape`, and `max_shape` arrays in the
manifest metadata or stop with `needs-shape-profile`.

Useful commands:

```bash
python3 -c 'import onnx,sys; m=onnx.load(sys.argv[1]); onnx.checker.check_model(m)' model.onnx
trtexec --onnx=model.onnx --minShapes=images:1x3x224x224 \
  --optShapes=images:1x3x224x224 --maxShapes=images:8x3x224x224 \
  --saveEngine=model.engine --fp16
```

### PyTorch, SafeTensors, and TorchScript

Use a caller-owned PyTorch environment and an explicit deterministic wrapper
that returns stable tensor outputs. Call `torch.onnx.export` with a
representative input and record `input_names`, `output_names`, opset, dynamic
axes/shapes, dtype, and preprocessing in the manifest. Record that script as
`export_wrapper`; command emission fails closed when no wrapper is supplied.
Do not export a training module with losses, random augmentation, or
postprocessing still attached.

Do not invoke Optimum or `optimum-cli`. Use the shared Python environment and
the explicit `torch.onnx.export` wrapper. Preserve the first exporter failure,
permit at most three total bounded operator repairs, and stop `BLOCKED` when
the wrapper cannot produce a checked ONNX graph.

SafeTensors is a weight container, not a graph. It needs the matching model
class/config and a deterministic forward wrapper. A file ending in `.bin` is
not automatically a PyTorch model: use its metadata rather than extension
guessing.

TorchScript is also not a native `nvinfer` input. Treat it as a source for a
wrapper export and compare outputs before proceeding.

### TensorFlow, Keras, and TFLite

`tf2onnx` can convert SavedModel/Keras/TFLite paths, for example:

```bash
python -m tf2onnx.convert --saved-model ./saved_model --output model.onnx --opset 17
python -m tf2onnx.convert --tflite ./model.tflite --output model.onnx --opset 17
```

These paths are experimental because layout, quantization, resize semantics,
custom ops, and postprocessing often differ. Preserve quantization metadata and
validate representative outputs. A TFLite file is never passed directly to
`nvinfer`.

### Rejected/other formats

Caffe/UFF import is rejected for current DeepStream releases. Paddle, JAX,
MXNet, Core ML, OpenVINO IR, GGUF, and proprietary formats require a vendor or
framework export to ONNX/TensorRT plus the same gates; the skill must not invent
a conversion command for them.

## Conversion records

Every conversion record must include exporter/library versions, opset,
precision/quantization, stable example-input identity, output names, command,
log, and validation result. Preserve caller-provided source or fixture hashes as
optional provenance; the skill does not compute or match them. Store conversions
beside the model and never replace the raw source.
