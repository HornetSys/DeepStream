<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NvDsTextEmbedder GStreamer Plugin

A DeepStream-compatible GStreamer Python plugin that generates text embeddings using the **SigLIP 2 Giant** model from Google (default) or, optionally, **DFN5B-CLIP** from Apple. This plugin is designed to work alongside `nvdsvisionencoder` for text-to-image search applications.

## Overview

The `nvdstextembedder` plugin generates text embeddings that can be used for:

1. **Text-to-image search**: Match text queries against vision embeddings from `nvdsvisionencoder`
2. **Person attribute search**: Search for people by description (e.g., "person wearing red shirt")
3. **Zero-shot classification**: Classify objects using natural language descriptions
4. **Semantic search**: Find similar concepts across text and images

### Supported Text Encoders

The encoder is selected with the `model-name` property. The default is `siglip2-giant`.

| `model-name` | Encoder | Embedding dim | Extra Python packages |
|--------------|---------|---------------|-----------------------|
| `siglip2-giant` (default; any name containing `giant`) | SigLIP 2 Giant (`google/siglip2-giant-opt-patch16-384`), PyTorch | 1536 | `transformers`, `sentencepiece` |
| `dfn5b-clip` (any name not matched by the rows below) | DFN5B-CLIP ViT-H-14-378 (`apple/DFN5B-CLIP-ViT-H-14-378`), OpenCLIP | 1024 | `open-clip-torch` |
| `siglip2-onnx` | SigLIP 2 text encoder, ONNX Runtime | model-dependent (taken from the ONNX model) | `onnxruntime-gpu`, `transformers` |
| name containing `dfnclip`, `dfn-clip`, `dfn_clip`, `radio-clip` or `radio_clip` | DFN-CLIP text encoder, ONNX Runtime | model-dependent (taken from the ONNX model) | `onnxruntime-gpu`, `transformers` |

The two ONNX encoders also require the `onnx-model-path` (ONNX model file) and `tokenizer-dir` (directory containing `tokenizer.json`) properties; without them no encoder is created. See [ONNX Encoders](#onnx-encoders-optional).

### SigLIP 2 Giant Model (default)

SigLIP 2 Giant from Google provides:
- Text embeddings aligned with the SigLIP 2 vision space
- 1536-dimensional, L2-normalized embedding vectors

Reference: [SigLIP 2 Giant on HuggingFace](https://huggingface.co/google/siglip2-giant-opt-patch16-384)

### DFN5B-CLIP Model (optional)

DFN5B-CLIP (Data Filtering Network 5B CLIP) from Apple provides:
- High-quality text embeddings aligned with CLIP vision space
- 1024-dimensional embedding vectors
- Trained on 5B image-text pairs with advanced data filtering

Select it with `model-name=dfn5b-clip`.

Reference: [DFN5B-CLIP on HuggingFace](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378)

### ONNX Encoders (optional)

Use these when you have an exported ONNX text encoder and want to run it with ONNX Runtime instead of PyTorch. Model names are matched case-insensitively:

- `model-name=siglip2-onnx` (or `siglip2_onnx`): SigLIP 2 ONNX text encoder
- `model-name` containing `dfnclip`, `dfn-clip`, `dfn_clip`, `radio-clip` or `radio_clip`: DFN-CLIP ONNX text encoder

Install the extra packages in the same (activated) environment:

```bash
pip install onnxruntime-gpu transformers
```

Then set the model and tokenizer locations on the element:

```python
embedder.set_property("model-name", "siglip2-onnx")
embedder.set_property("onnx-model-path", "/path/to/text_encoder.onnx")
embedder.set_property("tokenizer-dir", "/path/to/tokenizer_dir")  # contains tokenizer.json
```

The embedding dimension is determined by the ONNX model you supply.

## Requirements

### System Requirements
- Python 3.8+
- CUDA-enabled GPU (recommended)

### Python Dependencies

On Ubuntu 24.04, system-wide `pip install` fails with `externally-managed-environment` (PEP 668). Use a virtual environment that can still see the system `gi`/`Gst` bindings:

```bash
# GStreamer and Python bindings (installed system-wide via apt)
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    python3-gst-1.0 \
    python3-venv \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-python3-plugin-loader

# Virtual environment with access to system packages (gi/Gst)
python3 -m venv --system-site-packages ~/nvdstextembedder-venv
source ~/nvdstextembedder-venv/bin/activate

# Default encoder (SigLIP 2 Giant)
pip install torch numpy transformers sentencepiece

# Optional: DFN5B-CLIP encoder (model-name=dfn5b-clip)
pip install open-clip-torch
```

**Expected startup warning.** With the default encoder the plugin may print `Warning: onnxruntime not available. DFN-CLIP ONNX encoder disabled.` This is harmless: `onnxruntime-gpu` is only needed for the ONNX encoders (see [ONNX Encoders](#onnx-encoders-optional)). The default `siglip2-giant` and optional `dfn5b-clip` encoders work without it.

**Jetson (JetPack / L4T).** On aarch64 a plain `pip install torch` resolves to the PyPI wheel, which pulls several GB of bundled CUDA 13 libraries (cuDNN, NCCL, NVSHMEM, ...). Make sure the device has enough free disk space (plan for roughly 10 GB or more including model weights), or install the PyTorch build that matches your JetPack/L4T release instead of the PyPI wheel, then install the remaining packages without re-resolving torch:

```bash
pip install numpy transformers sentencepiece
```

Activate the environment in every shell that runs the plugin. The model weights are downloaded from HuggingFace on first use, so the first run needs network access.

## Installation

1. Clone or copy the plugin to your system:
```bash
git clone <repository> /path/to/gst-nvdstextembedder
```

2. Set the plugin path (GStreamer Python loader looks for `python/` subdirectory):
```bash
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/path/to/gst-nvdstextembedder
```

3. Verify plugin registration:
```bash
gst-inspect-1.0 nvdstextembedder
```

## Usage

### Python API (Recommended)

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import json
import numpy as np

Gst.init(None)

# Create text embedder element
embedder = Gst.ElementFactory.make("nvdstextembedder", "text-embedder")
embedder.set_property("model-name", "siglip2-giant")  # default; 1536-dim
embedder.set_property("device", "cuda")
embedder.set_property("verbose", True)

# Set texts to encode
texts = [
    "person wearing red shirt",
    "man with blue jacket",
    "woman carrying handbag"
]
embedder.set_property("texts", json.dumps(texts))

# Create a minimal pipeline to trigger model loading
pipeline = Gst.Pipeline.new("text-embed-pipeline")
source = Gst.ElementFactory.make("videotestsrc", "source")
sink = Gst.ElementFactory.make("fakesink", "sink")

source.set_property("num-buffers", 1)

pipeline.add(source)
pipeline.add(embedder)
pipeline.add(sink)

source.link(embedder)
embedder.link(sink)

# Run pipeline to trigger embedding computation
pipeline.set_state(Gst.State.PLAYING)

# Wait for model to load and embeddings to compute
import time
# (the first run downloads the model weights, which can take several minutes)
while not (embedder.get_property("model-loaded")
           and json.loads(embedder.get_property("embeddings"))):
    time.sleep(0.5)

# Get embeddings
embeddings_json = embedder.get_property("embeddings")
embeddings = np.array(json.loads(embeddings_json))

print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")

# Clean up
pipeline.set_state(Gst.State.NULL)
```

### Using the Embedding Query (On-Demand Encoding)

The plugin supports a custom GStreamer "embedding" query for on-demand text encoding. This is useful when you want to dynamically encode text without modifying properties. Results are returned **synchronously** in the query structure.

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import time

# Import query helper functions
from nvdstextembedder import (
    create_embedding_query,
    create_embedding_query_single,
    parse_embedding_query_result
)

Gst.init(None)

# ... (create pipeline with embedder element) ...

# Wait for model to load
while not embedder.get_property("model-loaded"):
    time.sleep(0.5)

# Create and send embedding query
query = create_embedding_query(["person in red clothing", "blue vehicle"])

# Send query to the element's src pad
src_pad = embedder.get_static_pad("src")
query_handled = src_pad.query(query)

# Parse result from the query structure (synchronous)
result = parse_embedding_query_result(query)

if result['success']:
    embeddings = np.array(result['embeddings'])
    print(f"Generated {result['num_embeddings']} embeddings")
    print(f"Embedding dimension: {result['embedding_dim']}")
    print(f"Embeddings shape: {embeddings.shape}")
else:
    print(f"Query failed: {result['error']}")
```

#### Query Result Structure

The `parse_embedding_query_result()` function returns a dictionary with:

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether encoding succeeded |
| `embeddings` | List[List[float]] | The embedding vectors |
| `num_embeddings` | int | Number of embeddings generated |
| `embedding_dim` | int | Dimension of each embedding |
| `error` | str | Error message if failed |

### Using the Embedding Query from C/C++ Code

The custom "embedding" query can also be sent from C/C++ code in your DeepStream application or custom GStreamer plugin. This is useful for integrating text embedding generation into native DeepStream pipelines.

#### Creating and Sending the Query (C Code)

```c
#include <gst/gst.h>
#include <json-glib/json-glib.h>
#include <string.h>
#include <stdio.h>

/**
 * Create an embedding query with the given texts.
 * 
 * @param texts Array of text strings to encode
 * @param num_texts Number of texts in the array
 * @return GstQuery* Custom query (caller must unref)
 */
GstQuery* create_embedding_query(const gchar** texts, guint num_texts) {
    GstStructure *structure;
    JsonBuilder *builder;
    JsonGenerator *generator;
    gchar *json_str;
    GstQuery *query;
    
    /* Build JSON array of texts */
    builder = json_builder_new();
    json_builder_begin_array(builder);
    for (guint i = 0; i < num_texts; i++) {
        json_builder_add_string_value(builder, texts[i]);
    }
    json_builder_end_array(builder);
    
    /* Generate JSON string */
    generator = json_generator_new();
    json_generator_set_root(generator, json_builder_get_root(builder));
    json_str = json_generator_to_data(generator, NULL);
    
    /* Create query structure with name "embedding" */
    structure = gst_structure_new("embedding",
                                   "text", G_TYPE_STRING, json_str,
                                   NULL);
    
    /* Create custom query */
    query = gst_query_new_custom(GST_QUERY_CUSTOM, structure);
    
    /* Cleanup */
    g_free(json_str);
    g_object_unref(generator);
    g_object_unref(builder);
    
    return query;
}

/**
 * Parse the embedding query result.
 * 
 * @param query The query after being processed
 * @param success Output: whether encoding succeeded
 * @param embeddings Output: array of embeddings (caller must free)
 * @param num_embeddings Output: number of embeddings
 * @param embedding_dim Output: dimension of each embedding
 * @return TRUE if parsing succeeded
 */
gboolean parse_embedding_query_result(GstQuery *query,
                                       gboolean *success,
                                       gfloat ***embeddings,
                                       guint *num_embeddings,
                                       guint *embedding_dim) {
    const GstStructure *structure;
    const gchar *embeddings_json;
    JsonParser *parser;
    JsonNode *root;
    JsonArray *array;
    GError *error = NULL;
    
    structure = gst_query_get_structure(query);
    if (!structure) {
        return FALSE;
    }
    
    /* Get success flag */
    if (gst_structure_has_field(structure, "success")) {
        gst_structure_get_boolean(structure, "success", success);
    } else {
        *success = FALSE;
    }
    
    /* Get embedding dimension */
    if (gst_structure_has_field(structure, "embedding_dim")) {
        gint dim;
        gst_structure_get_int(structure, "embedding_dim", &dim);
        *embedding_dim = (guint)dim;
    } else {
        *embedding_dim = 0;
    }
    
    /* Get number of embeddings */
    if (gst_structure_has_field(structure, "num_embeddings")) {
        gint num;
        gst_structure_get_int(structure, "num_embeddings", &num);
        *num_embeddings = (guint)num;
    } else {
        *num_embeddings = 0;
    }
    
    /* Parse embeddings JSON */
    embeddings_json = gst_structure_get_string(structure, "embeddings");
    if (!embeddings_json || *num_embeddings == 0) {
        *embeddings = NULL;
        return TRUE;
    }
    
    /* Parse JSON array */
    parser = json_parser_new();
    if (!json_parser_load_from_data(parser, embeddings_json, -1, &error)) {
        g_warning("Failed to parse embeddings JSON: %s", error->message);
        g_error_free(error);
        g_object_unref(parser);
        return FALSE;
    }
    
    root = json_parser_get_root(parser);
    array = json_node_get_array(root);
    
    /* Allocate embeddings array */
    *embeddings = g_new(gfloat*, *num_embeddings);
    
    for (guint i = 0; i < *num_embeddings; i++) {
        JsonArray *emb_array = json_array_get_array_element(array, i);
        guint dim = json_array_get_length(emb_array);
        
        (*embeddings)[i] = g_new(gfloat, dim);
        for (guint j = 0; j < dim; j++) {
            (*embeddings)[i][j] = (gfloat)json_array_get_double_element(emb_array, j);
        }
    }
    
    g_object_unref(parser);
    return TRUE;
}

/**
 * Example usage in a DeepStream application or probe callback.
 */
void example_send_embedding_query(GstElement *text_embedder) {
    const gchar *texts[] = {
        "person wearing red shirt",
        "man with blue jacket",
        "woman carrying handbag"
    };
    guint num_texts = 3;
    
    GstQuery *query;
    GstPad *src_pad;
    gboolean ret;
    
    /* Create the embedding query */
    query = create_embedding_query(texts, num_texts);
    
    /* Send query to the element's src pad */
    src_pad = gst_element_get_static_pad(text_embedder, "src");
    ret = gst_pad_query(src_pad, query);
    gst_object_unref(src_pad);
    
    if (ret) {
        gboolean success;
        gfloat **embeddings = NULL;
        guint num_embeddings, embedding_dim;
        
        /* Parse the result */
        if (parse_embedding_query_result(query, &success, &embeddings,
                                          &num_embeddings, &embedding_dim)) {
            if (success) {
                g_print("Generated %u embeddings with dimension %u\n",
                        num_embeddings, embedding_dim);
                
                /* Use the embeddings for similarity search, etc. */
                for (guint i = 0; i < num_embeddings; i++) {
                    g_print("Embedding %u: [%.4f, %.4f, %.4f, ...]\n",
                            i, embeddings[i][0], embeddings[i][1], embeddings[i][2]);
                }
                
                /* Free embeddings */
                for (guint i = 0; i < num_embeddings; i++) {
                    g_free(embeddings[i]);
                }
                g_free(embeddings);
            } else {
                g_warning("Embedding query failed");
            }
        }
    } else {
        g_warning("Failed to send embedding query");
    }
    
    gst_query_unref(query);
}
```

#### Query Structure Reference

The embedding query uses the following GstStructure format:

**Request (sent to element):**
```
embedding, text=(string)"[\"query1\", \"query2\"]"
```

**Response (after query is processed):**
```
embedding, text=(string)"[\"query1\", \"query2\"]", 
           success=(boolean)true, 
           embeddings=(string)"[[0.1, 0.2, ...], [0.3, 0.4, ...]]",
           num_embeddings=(int)2,
           embedding_dim=(int)1536
```

#### Compilation

When compiling C code that uses the embedding query, link against:
- `gstreamer-1.0`
- `json-glib-1.0`

```bash
gcc -o my_app my_app.c \
    $(pkg-config --cflags --libs gstreamer-1.0 json-glib-1.0)
```

#### Thread Safety

The embedding query is thread-safe and can be sent from:
- Pad probe callbacks
- Bus message handlers
- Separate application threads
- Other GStreamer plugins

Note: The first query may have higher latency as it triggers model loading. Subsequent queries will be faster.

### Using with nvdsvisionencoder for Text-to-Image Search

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import json
import numpy as np

Gst.init(None)

# Create text embedder for queries
text_embedder = Gst.ElementFactory.make("nvdstextembedder", "text-embedder")
text_embedder.set_property("model-name", "siglip2-giant")  # must match the vision encoder
text_embedder.set_property("device", "cuda")

# Set search queries
queries = ["person in red clothing", "blue car"]
text_embedder.set_property("texts", json.dumps(queries))

# ... (run pipeline to get embeddings)

# Get text embeddings
text_embeddings = np.array(json.loads(text_embedder.get_property("embeddings")))

# Vision embeddings from nvdsvisionencoder would be obtained from
# NVDS_TRACKER_OBJ_REID_META attached to detected objects

# Compute similarity (dot product since embeddings are normalized)
# similarity = np.dot(text_embeddings, vision_embeddings.T)
```

### GStreamer Pipeline (Basic)

```bash
# Set plugin path
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/path/to/gst-nvdstextembedder

# Run pipeline with text embedder
gst-launch-1.0 \
    videotestsrc num-buffers=10 ! \
    nvdstextembedder texts='["query1", "query2"]' verbose=true ! \
    fakesink
```

## Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `texts` | string | `[]` | JSON array of texts to encode |
| `embeddings` | string (read-only) | `[]` | JSON array of computed embeddings |
| `batch-size` | int | `64` | Maximum batch size for inference (1-128) |
| `verbose` | bool | `False` | Enable verbose logging |
| `device` | string | `cuda` | Inference device (`cuda` or `cpu`) |
| `use-fp16` | bool | `True` | Enable FP16 inference for faster processing |
| `model-name` | string | `siglip2-giant` | Encoder selection: `siglip2-giant` (1536-dim), `dfn5b-clip` (1024-dim), `siglip2-onnx` or a `dfnclip`/`dfn-clip`/`radio-clip` name (ONNX) |
| `onnx-model-path` | string | `""` | Path to the ONNX text encoder file (ONNX encoders only) |
| `tokenizer-dir` | string | `""` | Directory containing `tokenizer.json` (ONNX encoders only) |
| `embedding-dim` | int (read-only) | `1536` | Model embedding dimension (depends on `model-name`) |
| `num-texts` | int (read-only) | `0` | Number of texts currently loaded |
| `model-loaded` | bool (read-only) | `False` | Whether the model is loaded and ready |
| `last-query-result` | string (read-only) | `{}` | JSON string of last embedding query result |

## API Reference

### Encoder Classes

Internal classes for model management. `SigLIP2GiantTextEncoder` is the default; `DFN5BCLIPEncoder` has the same interface:

```python
class SigLIP2GiantTextEncoder:
    def __init__(self, device: str = "cuda", use_fp16: bool = True)
    def load_model(self) -> bool
    def encode_texts(self, texts: List[str]) -> np.ndarray
    def encode_single(self, text: str) -> np.ndarray
    
    @property
    def is_initialized(self) -> bool
    @property
    def embedding_dim(self) -> int
```

### NvDsTextEmbedder Element

Additional Python methods available on the element:

```python
# Get embeddings as Python list
embeddings = embedder.get_embeddings()

# Get embeddings as numpy array
embeddings_np = embedder.get_embeddings_numpy()

# Directly encode texts (bypasses JSON serialization)
embeddings = embedder.encode_texts(["text1", "text2"])
```

### Query Helper Functions

Helper functions for creating, sending, and parsing embedding queries:

```python
from nvdstextembedder import (
    create_embedding_query,
    create_embedding_query_single,
    parse_embedding_query_result
)

# Create a query for multiple texts
query = create_embedding_query(["text1", "text2", "text3"])

# Create a query for a single text
query = create_embedding_query_single("single text query")

# Send the query to the element
pad = embedder.get_static_pad("src")
pad.query(query)

# Parse result from the query structure (synchronous)
result = parse_embedding_query_result(query)

# Parse the result
result = parse_embedding_query_result(query)
# result = {
#     'success': True,
#     'embeddings': [[...], [...], [...]],
#     'num_embeddings': 3,
#     'embedding_dim': 1536,
#     'error': None
# }
```

## File Structure

```
gst-nvdstextembedder/
├── python/
│   └── nvdstextembedder.py    # Main plugin implementation
└── README.md                   # This file
```

## Testing

Verify plugin registration and the default encoder:

```bash
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD
gst-inspect-1.0 nvdstextembedder        # model-name default: "siglip2-giant", embedding-dim: 1536
```

Then run the Python API example from the Usage section; `model-loaded` becomes `True` once the model is downloaded and loaded.

## Performance Tips

### GPU Memory
- SigLIP 2 Giant and DFN5B-CLIP ViT-H-14 each need a few GB of GPU memory; the first run also downloads the weights
- Enable FP16 (`use-fp16=true`) to reduce memory usage

### Batch Size
- Increase `batch-size` for better GPU utilization when encoding many texts
- Default of 64 works well for most cases

### Caching
- Embeddings are computed once when `texts` property is set
- Retrieve via `embeddings` property without recomputation

## Troubleshooting

### Plugin Not Found
```bash
# Check plugin path
echo $GST_PLUGIN_PATH

# Set correct path
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/path/to/gst-nvdstextembedder

# Check for syntax errors
python3 -m py_compile python/nvdstextembedder.py
```

### Model Loading Errors
If `model-loaded` never becomes `True`, check the plugin log for `transformers not available` or `sentencepiece` warnings and install the packages listed under Python Dependencies in the same (activated) environment.

```bash
# Default encoder (SigLIP 2 Giant)
python3 -c "
from transformers import AutoProcessor, AutoModel
AutoProcessor.from_pretrained('google/siglip2-giant-opt-patch16-384', use_fast=False)
print('SigLIP2 tokenizer loaded successfully')
"

# DFN5B-CLIP encoder (model-name=dfn5b-clip)
python3 -c "
import open_clip
model, _ = open_clip.create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-378')
print('Model loaded successfully')
"
```

### ONNX Encoder Not Created
If `siglip2-onnx` or a DFN-CLIP model name is selected but no encoder loads, check the plugin log for:
- `onnx-model-path and/or tokenizer-dir not set`: set both properties
- `Missing packages for ... ONNX`: `pip install onnxruntime-gpu transformers` in the activated environment

### CUDA Out of Memory
- Set `device=cpu` for CPU-only inference
- Reduce `batch-size`
- Check other GPU processes with `nvidia-smi`

### No Embeddings Generated
- Enable `verbose=true` for debugging
- Ensure `texts` property is set before checking `embeddings`
- Wait for `model-loaded` property to become `True`

## Integration with nvdsvisionencoder

The text embeddings from `nvdstextembedder` are designed to be compared with vision embeddings from `nvdsvisionencoder`:

```python
import numpy as np

# Text embeddings from nvdstextembedder
text_emb = np.array(json.loads(text_embedder.get_property("embeddings")))

# Vision embeddings from nvdsvisionencoder (via NVDS_TRACKER_OBJ_REID_META)
# ... retrieved from object metadata ...

# Compute cosine similarity (embeddings are already L2-normalized)
similarities = np.dot(text_emb, vision_emb.T)

# Find best matches
best_matches = np.argmax(similarities, axis=1)
```

## Example Applications

### Person Attribute Search
Search for people matching text descriptions in video streams:
```python
queries = [
    "person wearing red shirt",
    "man with blue jeans",
    "woman carrying handbag"
]
embedder.set_property("texts", json.dumps(queries))
# Compare with vision embeddings from detected persons
```

### Zero-Shot Classification
Classify detected objects using natural language:
```python
classes = [
    "car",
    "truck",
    "motorcycle",
    "bicycle",
    "pedestrian"
]
embedder.set_property("texts", json.dumps(classes))
# Match vision embeddings to class embeddings
```

### Visual Search
Search for specific objects by description:
```python
search_query = ["red sports car parked on street"]
embedder.set_property("texts", json.dumps(search_query))
# Find objects with highest similarity to query
```

## References

- [SigLIP 2 Giant Model](https://huggingface.co/google/siglip2-giant-opt-patch16-384)
- [DFN5B-CLIP Model](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378)
- [OpenCLIP Library](https://github.com/mlfoundations/open_clip)
- [DeepStream SDK Documentation](https://nvidia-deepstream.docs.buildwithfern.com/deepstream/9.2)
- [GStreamer Python Plugin Development](https://gstreamer.freedesktop.org/documentation/plugin-development/index.html)

## License

See repository license file.

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.
