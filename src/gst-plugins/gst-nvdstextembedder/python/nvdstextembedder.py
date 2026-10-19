#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
GStreamer Python Plugin: nvdstextembedder

A DeepStream-compatible GStreamer plugin that generates text embeddings using
the DFN5B-CLIP model from Apple. This plugin is designed to work alongside
nvdsvisionencoder for text-to-image search applications.

Features:
    - Generates text embeddings using DFN5B-CLIP ViT-H-14 model
    - Batch processing for efficient GPU inference
    - JSON-based text input and embedding output
    - Custom GStreamer "embedding" query for on-demand text encoding
    - Compatible with nvdsvisionencoder for text-to-image similarity search

Requirements:
    - open-clip-torch library (pip install open-clip-torch)
    - torch (PyTorch)
    - CUDA-enabled GPU (recommended)

Usage:
    # GStreamer Python loader automatically looks for python/ subdirectory
    export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/path/to/gst-nvdstextembedder
    
    # In a pipeline (typically used for text query encoding):
    ... ! nvdstextembedder texts='["person in red", "blue car"]' ! ...

    # Or via Python API:
    embedder = Gst.ElementFactory.make("nvdstextembedder", "text-embedder")
    embedder.set_property("texts", '["query1", "query2"]')

    # Using the custom embedding query (synchronous):
    from nvdstextembedder import create_embedding_query, parse_embedding_query_result
    
    query = create_embedding_query(["person wearing red", "blue car"])
    pad = embedder.get_static_pad("src")
    pad.query(query)  # Results returned synchronously in query structure
    result = parse_embedding_query_result(query)
    embeddings = result['embeddings']  # List of embedding vectors

Reference:
    - DFN5B-CLIP: https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378
    - open_clip: https://github.com/mlfoundations/open_clip
"""

import gi
import sys
import json
import ctypes
import threading
import queue
import uuid
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')

from gi.repository import Gst, GObject, GstBase

Gst.init(None)

# Custom query type for text embedding
# Using GST_QUERY_CUSTOM as base for application-specific queries
EMBEDDING_QUERY_NAME = "embedding"
# =============================================================================
# Low-level ctypes helpers for GStreamer structure modification
# 
# Python GStreamer bindings have a limitation where writable_structure() returns
# a copy instead of a reference. We use ctypes to directly call C functions to
# properly modify query structures so results can be returned synchronously.
# =============================================================================

# Load GStreamer and GObject libraries
try:
    _libgst = ctypes.CDLL('libgstreamer-1.0.so.0')
    _libgobject = ctypes.CDLL('libgobject-2.0.so.0')
    _CTYPES_AVAILABLE = True
except OSError:
    _CTYPES_AVAILABLE = False
    print("[NvDsTextEmbedder] Warning: Could not load GStreamer libraries via ctypes")

if _CTYPES_AVAILABLE:
    # GValue structure for passing values to GStreamer
    class _GValue(ctypes.Structure):
        _fields_ = [
            ('g_type', ctypes.c_ulong),
            ('data', ctypes.c_uint64 * 2)
        ]
    
    # GType constants
    _G_TYPE_BOOLEAN = 5 << 2   # 20
    _G_TYPE_INT = 6 << 2       # 24  
    _G_TYPE_UINT64 = 11 << 2   # 44
    _G_TYPE_STRING = 16 << 2   # 64
    
    # Function prototypes
    _libgst.gst_query_writable_structure.argtypes = [ctypes.c_void_p]
    _libgst.gst_query_writable_structure.restype = ctypes.c_void_p
    
    _libgst.gst_structure_set_value.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_GValue)]
    _libgst.gst_structure_set_value.restype = None
    
    _libgobject.g_value_init.argtypes = [ctypes.POINTER(_GValue), ctypes.c_ulong]
    _libgobject.g_value_init.restype = ctypes.POINTER(_GValue)
    
    _libgobject.g_value_set_boolean.argtypes = [ctypes.POINTER(_GValue), ctypes.c_int]
    _libgobject.g_value_set_string.argtypes = [ctypes.POINTER(_GValue), ctypes.c_char_p]
    _libgobject.g_value_set_int.argtypes = [ctypes.POINTER(_GValue), ctypes.c_int]
    _libgobject.g_value_set_uint64.argtypes = [ctypes.POINTER(_GValue), ctypes.c_uint64]


def _set_query_structure_value(query: Gst.Query, field_name: str, value, value_type: str = None) -> bool:
    """
    Set a value on a query's structure using ctypes (workaround for Python bindings limitation).
    
    Args:
        query: GStreamer query object
        field_name: Name of the field to set
        value: Value to set (bool, int, or str)
        value_type: Optional type hint ('uint64' for unsigned 64-bit int)
        
    Returns:
        True if successful, False otherwise
    """
    if not _CTYPES_AVAILABLE:
        return False
    
    try:
        # Get query pointer (PyGObject: hash() often returns the C pointer for GObject wrappers)
        query_ptr = getattr(query, "_get_c_ptr", None)
        if callable(query_ptr):
            query_ptr = query_ptr()
        else:
            query_ptr = hash(query)
        if isinstance(query_ptr, int):
            query_ptr = ctypes.c_void_p(query_ptr)
        struct_ptr = _libgst.gst_query_writable_structure(query_ptr)
        
        if struct_ptr is None or (isinstance(struct_ptr, ctypes.c_void_p) and not struct_ptr.value) or struct_ptr == 0:
            return False
        
        field_bytes = field_name.encode('utf-8')
        gvalue = _GValue()
        
        if isinstance(value, bool):
            _libgobject.g_value_init(ctypes.byref(gvalue), _G_TYPE_BOOLEAN)
            _libgobject.g_value_set_boolean(ctypes.byref(gvalue), 1 if value else 0)
        elif isinstance(value, int):
            if value_type == 'uint64':
                _libgobject.g_value_init(ctypes.byref(gvalue), _G_TYPE_UINT64)
                _libgobject.g_value_set_uint64(ctypes.byref(gvalue), ctypes.c_uint64(value))
            else:
                _libgobject.g_value_init(ctypes.byref(gvalue), _G_TYPE_INT)
                _libgobject.g_value_set_int(ctypes.byref(gvalue), value)
        elif isinstance(value, str):
            _libgobject.g_value_init(ctypes.byref(gvalue), _G_TYPE_STRING)
            _libgobject.g_value_set_string(ctypes.byref(gvalue), value.encode('utf-8'))
        else:
            return False
        
        _libgst.gst_structure_set_value(struct_ptr, field_bytes, ctypes.byref(gvalue))
        return True
        
    except Exception as e:
        print(f"[NvDsTextEmbedder] Error setting query structure value: {e}")
        return False

# Try to import ML libraries
try:
    import torch
    import torch.nn.functional as F
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[NvDsTextEmbedder] Warning: torch not available. Model inference disabled.")

# Try to import open_clip
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    print("[NvDsTextEmbedder] Warning: open_clip not available. Install with: pip install open-clip-torch")

# Try to import transformers (for SigLIP2 Giant text encoder and CLIPTokenizer)
try:
    from transformers import AutoModel, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[NvDsTextEmbedder] Warning: transformers not available. SigLIP2 Giant text encoder disabled. Install with: pip install transformers")

# Try to import ONNX Runtime (for DFN-CLIP ONNX inference)
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("[NvDsTextEmbedder] Warning: onnxruntime not available. DFN-CLIP ONNX encoder disabled. Install with: pip install onnxruntime-gpu")


def _apply_transformers_tokenizer_patch():
    """
    Fix AttributeError when loading SigLIP2 processor on transformers >= 4.45.
    TOKENIZER_MAPPING_NAMES.get(..., "") can return None for SigLIP2, so .replace() raises.
    Applied only when loading SigLIP2 Giant text encoder.
    """
    try:
        import transformers.models.auto.tokenization_auto as tokenization_auto
        if getattr(tokenization_auto, "_tokenizer_patch_applied", False):
            return
        if getattr(tokenization_auto, "TOKENIZER_MAPPING_NAMES", None) is None:
            return
        class _SafeMapping(dict):
            def get(self, key, default=None):
                v = super().get(key, default)
                if v is None and default == "":
                    return ""
                return v
        if isinstance(tokenization_auto.TOKENIZER_MAPPING_NAMES, dict):
            tokenization_auto.TOKENIZER_MAPPING_NAMES = _SafeMapping(
                tokenization_auto.TOKENIZER_MAPPING_NAMES
            )
            tokenization_auto._tokenizer_patch_applied = True
            print("[NvDsTextEmbedder] Applied transformers tokenizer patch (SigLIP2 compatibility)")
    except Exception as e:
        print(f"[NvDsTextEmbedder] Note: Could not apply tokenizer patch: {e}")


# Define capabilities for the pads
# This is a passthrough element - it doesn't modify video data
CAPS = Gst.Caps.from_string(
    "video/x-raw(memory:NVMM), format={NV12, RGBA, I420, RGB, BGR}; "
    "video/x-raw, format={RGB, BGR, RGBA, NV12, I420}; "
    "ANY"
)

# Pad templates
SINK_TEMPLATE = Gst.PadTemplate.new(
    "sink",
    Gst.PadDirection.SINK,
    Gst.PadPresence.ALWAYS,
    CAPS
)

SRC_TEMPLATE = Gst.PadTemplate.new(
    "src",
    Gst.PadDirection.SRC,
    Gst.PadPresence.ALWAYS,
    CAPS
)


class DFN5BCLIPEncoder:
    """
    DFN5B-CLIP Text Encoder wrapper for generating text embeddings.
    
    This class handles model loading and text encoding using Apple's
    DFN5B-CLIP model via the open_clip library.
    
    The DFN5B-CLIP model produces 1024-dimensional embeddings that are
    compatible with CLIP-style vision encoders for text-to-image search.
    
    Thread Safety:
        This class is thread-safe. All inference operations are protected
        by a lock to prevent concurrent access issues when called from
        C GStreamer pipelines.
    """
    
    # Model configuration
    MODEL_NAME = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"
    TOKENIZER_NAME = "ViT-H-14"
    EMBEDDING_DIM = 1024  # DFN5B-CLIP ViT-H-14 embedding dimension
    
    def __init__(self, device: str = "cuda", use_fp16: bool = True):
        """
        Initialize the DFN5B-CLIP encoder.
        
        Args:
            device: Device to run inference on ("cuda" or "cpu")
            use_fp16: Enable FP16 inference for faster processing (default: True)
        """
        self.model = None
        self.tokenizer = None
        self.device = device
        self.embedding_dim = self.EMBEDDING_DIM
        self._initialized = False
        self.use_fp16 = use_fp16 and device == "cuda"
        
        # Cached device for fast inference
        self._cached_device = None
        self._cached_dtype = None
        
        # Thread lock for safe concurrent access from C GStreamer pipelines
        self._lock = threading.Lock()
    
    def load_model(self) -> bool:
        """
        Load the DFN5B-CLIP model and tokenizer.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not ML_AVAILABLE or not OPENCLIP_AVAILABLE:
            print("[DFN5BCLIPEncoder] ML libraries not available")
            return False
        
        try:
            print(f"[DFN5BCLIPEncoder] Loading model: {self.MODEL_NAME}")
            print(f"[DFN5BCLIPEncoder] Device: {self.device}, FP16: {self.use_fp16}")
            
            # Load model from HuggingFace hub via open_clip
            self.model, _ = open_clip.create_model_from_pretrained(self.MODEL_NAME)
            self.tokenizer = open_clip.get_tokenizer(self.TOKENIZER_NAME)
            
            # Move to device and set eval mode
            self.model = self.model.eval().to(self.device)
            
            # Cache device and dtype for fast inference
            self._cached_device = torch.device(self.device)
            self._cached_dtype = torch.float16 if self.use_fp16 else torch.float32
            
            # Determine actual embedding dimension from model
            self._determine_embedding_dim()
            
            # Synchronize CUDA after model loading
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            self._initialized = True
            print(f"[DFN5BCLIPEncoder] Model loaded successfully. Embedding dim: {self.embedding_dim}")
            
            # Warmup the model
            self._warmup()
            
            return True
            
        except Exception as e:
            print(f"[DFN5BCLIPEncoder] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            return False
    
    def _determine_embedding_dim(self):
        """Determine the embedding dimension by running a dummy inference."""
        try:
            dummy_text = ["test"]
            tokens = self.tokenizer(dummy_text).to(self._cached_device)
            
            with torch.no_grad():
                output = self.model.encode_text(tokens)
            
            self.embedding_dim = output.shape[-1]
            print(f"[DFN5BCLIPEncoder] Detected embedding dim: {self.embedding_dim}")
            
        except Exception as e:
            print(f"[DFN5BCLIPEncoder] Could not determine embedding dim: {e}")
            # Use default
            self.embedding_dim = self.EMBEDDING_DIM
    
    def _warmup(self, num_warmup: int = 3):
        """
        Warmup the model with dummy inference to compile CUDA kernels.
        """
        print(f"[DFN5BCLIPEncoder] Warming up model ({num_warmup} iterations)...")
        try:
            dummy_texts = ["warmup text for model initialization"]
            
            for i in range(num_warmup):
                _ = self.encode_texts(dummy_texts)
            
            # Synchronize to ensure warmup is complete
            if self._cached_device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            print(f"[DFN5BCLIPEncoder] Warmup complete")
            
        except Exception as e:
            print(f"[DFN5BCLIPEncoder] Warmup failed (non-critical): {e}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if the model is initialized and ready for inference."""
        return self._initialized and self.model is not None and self.tokenizer is not None
    
    def encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Encode a batch of texts into embeddings.
        
        This method is thread-safe and can be called from C GStreamer pipelines.
        CUDA operations are synchronized before returning to ensure the numpy
        array data is fully available.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of embeddings (N, embedding_dim) or None on error
        """
        if not self.is_initialized:
            print("[DFN5BCLIPEncoder] Model not initialized")
            return None
        
        if not texts:
            return np.array([], dtype=np.float32)
        
        # Use lock to ensure thread safety when called from C GStreamer pipelines
        with self._lock:
            try:
                # Tokenize texts
                tokens = self.tokenizer(texts).to(self._cached_device)
                
                # Get embeddings with optional autocast for FP16
                with torch.inference_mode():
                    if self.use_fp16:
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            features = self.model.encode_text(tokens)
                    else:
                        features = self.model.encode_text(tokens)
                    
                    # Normalize embeddings (L2 normalization)
                    features = F.normalize(features.float(), dim=-1)
                
                # CRITICAL: Synchronize CUDA before converting to numpy
                # This ensures all GPU operations complete before the data is accessed
                # Prevents segfaults when called from C GStreamer pipelines
                if self._cached_device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Convert to numpy - data is now safe to access from C code
                # Use .copy() to ensure we own the memory (not a view of GPU memory)
                embeddings_np = features.cpu().numpy().copy()
                
                return embeddings_np
                
            except Exception as e:
                print(f"[DFN5BCLIPEncoder] Error encoding texts: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    def encode_single(self, text: str) -> Optional[np.ndarray]:
        """
        Encode a single text into an embedding.
        
        Args:
            text: Text string to encode
            
        Returns:
            Numpy array embedding (embedding_dim,) or None on error
        """
        result = self.encode_texts([text])
        if result is not None and len(result) > 0:
            return result[0]
        return None


class SigLIP2GiantTextEncoder:
    """
    SigLIP 2 Giant text encoder (text-only, 1536-dim).
    Uses google/siglip2-giant-opt-patch16-384, 1536-dim; processor with use_fast=False;
    handles pooler_output / last_hidden_state and optional text_projection; L2-normalized.
    Thread-safe via internal lock.
    """
    MODEL_ID = "google/siglip2-giant-opt-patch16-384"
    EMBEDDING_DIM = 1536

    def __init__(self, device: str = "cuda", use_fp16: bool = True):
        self.model = None
        self.processor = None
        self.device = device
        self.embedding_dim = self.EMBEDDING_DIM
        self._initialized = False
        self.use_fp16 = use_fp16 and device == "cuda"
        self._cached_device = None
        self._lock = threading.Lock()
        self._max_length = 64

    def load_model(self) -> bool:
        if not ML_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            print("[SigLIP2GiantTextEncoder] ML or transformers not available")
            return False
        try:
            _apply_transformers_tokenizer_patch()
            print(f"[SigLIP2GiantTextEncoder] Loading model: {self.MODEL_ID}")
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_ID, trust_remote_code=True, use_fast=False
            )
            self.model = AutoModel.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            self.model = self.model.eval().to(self.device)
            self._cached_device = torch.device(self.device)
            try:
                self._max_length = self.model.config.text_config.max_position_embeddings
            except Exception:
                self._max_length = 64
            if self.device == "cuda":
                torch.cuda.synchronize()
            self._initialized = True
            print(f"[SigLIP2GiantTextEncoder] Model loaded. Embedding dim: {self.embedding_dim}, max_length: {self._max_length}")
            self._warmup()
            return True
        except ImportError as e:
            err_msg = str(e)
            if "sentencepiece" in err_msg.lower() or "SentencePiece" in err_msg:
                print("[SigLIP2GiantTextEncoder] SigLIP2 tokenizer requires sentencepiece. Install with: pip install sentencepiece")
            print(f"[SigLIP2GiantTextEncoder] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            return False
        except Exception as e:
            print(f"[SigLIP2GiantTextEncoder] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            return False

    def _warmup(self, num_warmup: int = 2):
        try:
            for _ in range(num_warmup):
                _ = self.encode_texts(["warmup"])
            if self._cached_device.type == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[SigLIP2GiantTextEncoder] Warmup failed (non-critical): {e}")

    @property
    def is_initialized(self) -> bool:
        return self._initialized and self.model is not None and self.processor is not None

    def encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.is_initialized or not texts:
            return np.array([], dtype=np.float32) if not texts else None
        with self._lock:
            try:
                inputs = self.processor(
                    text=texts,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self._max_length,
                    truncation=True,
                )
                inputs = {k: v.to(self._cached_device) for k, v in inputs.items()}
                with torch.inference_mode():
                    if self.use_fp16:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            out = self.model.get_text_features(**inputs)
                    else:
                        out = self.model.get_text_features(**inputs)
                    if isinstance(out, torch.Tensor):
                        features = out
                    else:
                        if hasattr(out, "pooler_output") and out.pooler_output is not None:
                            features = out.pooler_output
                        elif hasattr(out, "last_hidden_state"):
                            features = out.last_hidden_state.mean(dim=1)
                        else:
                            raise AttributeError(f"Cannot get embedding from model output: {type(out)}")
                        if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
                            features = self.model.text_projection(features)
                    features = F.normalize(features.float(), dim=-1)
                if self._cached_device.type == "cuda":
                    torch.cuda.synchronize()
                return features.cpu().numpy().copy()
            except Exception as e:
                print(f"[SigLIP2GiantTextEncoder] Error encoding: {e}")
                import traceback
                traceback.print_exc()
                return None

    def encode_single(self, text: str) -> Optional[np.ndarray]:
        result = self.encode_texts([text])
        if result is not None and len(result) > 0:
            return result[0]
        return None


class DFNClipONNXTextEncoder:
    """
    DFN-CLIP text encoder using ONNX Runtime for inference and a local
    CLIPTokenizer (from the RADIO-CLIP export).

    At load time the combined image+text CLIP ONNX graph is analysed with
    ``onnx.utils.Extractor`` and a **text-only** sub-model is extracted
    (input: ``input_ids`` -> output: ``text_embedding``).  This eliminates
    the image encoder entirely -- no dummy image tensor, no wasted compute.

    If extraction fails (e.g. ``onnx`` package missing), the full model is
    loaded as a fallback and a zero-filled image tensor is supplied at
    inference time.

    Thread-safe via internal lock.
    """

    EMBEDDING_DIM = 1024
    IMAGE_SIZE = 224
    MAX_LENGTH = 77

    def __init__(self, onnx_model_path: str, tokenizer_dir: str,
                 device: str = "cuda"):
        self.onnx_model_path = onnx_model_path
        self.tokenizer_dir = tokenizer_dir
        self.device = device
        self.embedding_dim = self.EMBEDDING_DIM
        self._initialized = False
        self._text_only = False
        self.session = None
        self.tokenizer = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def load_model(self) -> bool:
        if not ONNXRUNTIME_AVAILABLE:
            print("[DFNClipONNXTextEncoder] onnxruntime not available")
            return False
        if not TRANSFORMERS_AVAILABLE:
            print("[DFNClipONNXTextEncoder] transformers not available "
                  "(needed for tokenizer)")
            return False

        import os
        try:
            print(f"[DFNClipONNXTextEncoder] Loading ONNX model: "
                  f"{self.onnx_model_path}")
            print(f"[DFNClipONNXTextEncoder] Tokenizer dir  : "
                  f"{self.tokenizer_dir}")

            # --- tokenizer ---------------------------------------------------
            tokenizer_json = self._find_file("tokenizer.json")
            if not tokenizer_json:
                print("[DFNClipONNXTextEncoder] ERROR: tokenizer.json not "
                      f"found in {self.tokenizer_dir}")
                return False

            from transformers import PreTrainedTokenizerFast
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_json,
                model_max_length=self.MAX_LENGTH,
                bos_token="<|startoftext|>",
                eos_token="<|endoftext|>",
                pad_token="<|endoftext|>",
                unk_token="<|endoftext|>",
            )
            print(f"[DFNClipONNXTextEncoder] Tokenizer loaded "
                  f"(max_length={self.MAX_LENGTH})")

            # --- ONNX model --------------------------------------------------
            if not os.path.isfile(self.onnx_model_path):
                print(f"[DFNClipONNXTextEncoder] ERROR: ONNX model not "
                      f"found: {self.onnx_model_path}")
                return False

            providers: list = []
            if self.device == "cuda":
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers.append(("CUDAExecutionProvider", {}))
            providers.append(("CPUExecutionProvider", {}))

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

            # Try to extract text-only subgraph so the image encoder is
            # removed entirely and we never need to feed an image tensor.
            text_model_bytes = self._extract_text_submodel()

            if text_model_bytes is not None:
                self.session = ort.InferenceSession(
                    text_model_bytes,
                    sess_options=sess_opts,
                    providers=providers,
                )
                self._text_only = True
            else:
                self.session = ort.InferenceSession(
                    self.onnx_model_path,
                    sess_options=sess_opts,
                    providers=providers,
                )
                self._text_only = False

            active_provider = self.session.get_providers()[0]
            print(f"[DFNClipONNXTextEncoder] ONNX session created "
                  f"(provider={active_provider}, "
                  f"text_only={self._text_only})")

            self._determine_embedding_dim()
            self._initialized = True
            print(f"[DFNClipONNXTextEncoder] Ready. "
                  f"Embedding dim: {self.embedding_dim}")

            self._warmup()
            return True

        except Exception as e:
            print(f"[DFNClipONNXTextEncoder] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            return False

    # ------------------------------------------------------------------
    def _extract_text_submodel(self) -> Optional[bytes]:
        """
        Extract the text-encoder-only subgraph from the combined CLIP model.

        Uses ``onnx.utils.Extractor`` to keep only the path from
        ``input_ids`` to ``text_embedding``, discarding all image-encoder
        nodes and the ``image`` input.

        Returns serialised ONNX model bytes, or None on failure.
        """
        try:
            import onnx
            from onnx.utils import Extractor
        except ImportError:
            print("[DFNClipONNXTextEncoder] 'onnx' package not installed; "
                  "cannot extract text subgraph, will use full model")
            return None

        try:
            print("[DFNClipONNXTextEncoder] Extracting text-only subgraph "
                  "from combined CLIP model ...")
            model = onnx.load(self.onnx_model_path)
            extractor = Extractor(model)
            text_model = extractor.extract_model(
                input_names=["input_ids"],
                output_names=["text_embedding"],
            )
            model_bytes = text_model.SerializeToString()
            size_mb = len(model_bytes) / (1024 * 1024)
            print(f"[DFNClipONNXTextEncoder] Text-only subgraph extracted "
                  f"({size_mb:.1f} MB, image encoder removed)")
            del model, text_model
            return model_bytes
        except Exception as e:
            print(f"[DFNClipONNXTextEncoder] Could not extract text "
                  f"subgraph ({e}); falling back to full model")
            return None

    # ------------------------------------------------------------------
    def _find_file(self, suffix: str) -> Optional[str]:
        """Return the first file in *tokenizer_dir* whose name ends with *suffix*."""
        import os
        if not os.path.isdir(self.tokenizer_dir):
            return None
        for fname in os.listdir(self.tokenizer_dir):
            if fname.endswith(suffix):
                return os.path.join(self.tokenizer_dir, fname)
        return None

    def _determine_embedding_dim(self):
        for out in self.session.get_outputs():
            if out.name == "text_embedding":
                shape = out.shape
                if len(shape) >= 2 and isinstance(shape[-1], int):
                    self.embedding_dim = shape[-1]
                    print(f"[DFNClipONNXTextEncoder] Detected embedding "
                          f"dim from ONNX output: {self.embedding_dim}")
                return

    def _warmup(self, num_warmup: int = 2):
        print(f"[DFNClipONNXTextEncoder] Warming up "
              f"({num_warmup} iterations)...")
        try:
            for _ in range(num_warmup):
                _ = self.encode_texts(["warmup text"])
            print("[DFNClipONNXTextEncoder] Warmup complete")
        except Exception as e:
            print(f"[DFNClipONNXTextEncoder] Warmup failed "
                  f"(non-critical): {e}")

    # ------------------------------------------------------------------
    @property
    def is_initialized(self) -> bool:
        return (self._initialized
                and self.session is not None
                and self.tokenizer is not None)

    def encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.is_initialized:
            print("[DFNClipONNXTextEncoder] Model not initialized")
            return None
        if not texts:
            return np.array([], dtype=np.float32)

        with self._lock:
            try:
                encoded = self.tokenizer(
                    texts,
                    padding="max_length",
                    max_length=self.MAX_LENGTH,
                    truncation=True,
                    return_tensors="np",
                )
                input_ids = encoded["input_ids"].astype(np.int64)

                if self._text_only:
                    # Extracted subgraph: only input_ids needed
                    outputs = self.session.run(
                        ["text_embedding"],
                        {"input_ids": input_ids},
                    )
                else:
                    # Full model fallback: supply dummy image
                    batch_size = input_ids.shape[0]
                    dummy_image = np.zeros(
                        (batch_size, 3, self.IMAGE_SIZE, self.IMAGE_SIZE),
                        dtype=np.float32,
                    )
                    outputs = self.session.run(
                        ["text_embedding"],
                        {"image": dummy_image, "input_ids": input_ids},
                    )

                text_emb = outputs[0].astype(np.float32)

                norms = np.linalg.norm(text_emb, axis=-1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                text_emb = text_emb / norms

                return text_emb

            except Exception as e:
                print(f"[DFNClipONNXTextEncoder] Error encoding texts: {e}")
                import traceback
                traceback.print_exc()
                return None

    def encode_single(self, text: str) -> Optional[np.ndarray]:
        result = self.encode_texts([text])
        if result is not None and len(result) > 0:
            return result[0]
        return None


class SigLIP2ONNXTextEncoder:
    """
    SigLIP2 text encoder using ONNX Runtime for inference and a local
    GemmaTokenizer (SentencePiece-based, from the SigLIP2 export).

    Same architecture as DFNClipONNXTextEncoder but with:
      - GemmaTokenizer (SentencePiece) instead of CLIPTokenizer (BPE)
      - max_length=64 (vs 77 for CLIP)
      - ``attention_mask`` input required alongside ``input_ids``
      - image size 256x256 (vs 224x224)

    At load time the text-only sub-model is extracted from the combined
    ONNX graph (``{input_ids, attention_mask} -> text_embedding``),
    eliminating the image encoder entirely.

    Thread-safe via internal lock.
    """

    EMBEDDING_DIM = 1152
    IMAGE_SIZE = 256
    MAX_LENGTH = 64

    def __init__(self, onnx_model_path: str, tokenizer_dir: str,
                 device: str = "cuda"):
        self.onnx_model_path = onnx_model_path
        self.tokenizer_dir = tokenizer_dir
        self.device = device
        self.embedding_dim = self.EMBEDDING_DIM
        self._initialized = False
        self._text_only = False
        self.session = None
        self.tokenizer = None
        self._lock = threading.Lock()
        self._needs_attention_mask = False
        self._image_size = self.IMAGE_SIZE

    # ------------------------------------------------------------------
    def load_model(self) -> bool:
        if not ONNXRUNTIME_AVAILABLE:
            print("[SigLIP2ONNXTextEncoder] onnxruntime not available")
            return False
        if not TRANSFORMERS_AVAILABLE:
            print("[SigLIP2ONNXTextEncoder] transformers not available "
                  "(needed for tokenizer)")
            return False

        import os
        try:
            print(f"[SigLIP2ONNXTextEncoder] Loading ONNX model: "
                  f"{self.onnx_model_path}")
            print(f"[SigLIP2ONNXTextEncoder] Tokenizer dir  : "
                  f"{self.tokenizer_dir}")

            # --- tokenizer ---------------------------------------------------
            tokenizer_json = self._find_file("tokenizer.json")
            if not tokenizer_json:
                print("[SigLIP2ONNXTextEncoder] ERROR: tokenizer.json not "
                      f"found in {self.tokenizer_dir}")
                return False

            from transformers import PreTrainedTokenizerFast
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_json,
                model_max_length=self.MAX_LENGTH,
                bos_token="<bos>",
                eos_token="<eos>",
                pad_token="<pad>",
                unk_token="<unk>",
                padding_side="right",
            )
            print(f"[SigLIP2ONNXTextEncoder] Tokenizer loaded "
                  f"(max_length={self.MAX_LENGTH})")

            # --- ONNX model --------------------------------------------------
            if not os.path.isfile(self.onnx_model_path):
                print(f"[SigLIP2ONNXTextEncoder] ERROR: ONNX model not "
                      f"found: {self.onnx_model_path}")
                return False

            available = ort.get_available_providers()
            providers: list = []
            if self.device == "cuda" and "CUDAExecutionProvider" in available:
                providers.append(("CUDAExecutionProvider", {}))
            providers.append(("CPUExecutionProvider", {}))

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

            text_model_bytes = self._extract_text_submodel()

            if text_model_bytes is not None:
                self.session = ort.InferenceSession(
                    text_model_bytes,
                    sess_options=sess_opts,
                    providers=providers,
                )
                self._text_only = True
            else:
                self.session = ort.InferenceSession(
                    self.onnx_model_path,
                    sess_options=sess_opts,
                    providers=providers,
                )
                self._text_only = False

            # Detect whether attention_mask is a model input
            input_names = {inp.name for inp in self.session.get_inputs()}
            self._needs_attention_mask = "attention_mask" in input_names

            # Infer image size from model signature for full-model fallback.
            image_input = next((inp for inp in self.session.get_inputs()
                                if inp.name == "image"), None)
            if image_input is not None and len(image_input.shape) >= 4:
                h = image_input.shape[2]
                w = image_input.shape[3]
                if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                    self._image_size = h if h == w else max(h, w)

            active_provider = self.session.get_providers()[0]
            print(f"[SigLIP2ONNXTextEncoder] ONNX session created "
                  f"(provider={active_provider}, "
                  f"text_only={self._text_only}, "
                  f"attention_mask={self._needs_attention_mask}, "
                  f"image_size={self._image_size})")

            self._determine_embedding_dim()
            self._initialized = True
            print(f"[SigLIP2ONNXTextEncoder] Ready. "
                  f"Embedding dim: {self.embedding_dim}")

            self._warmup()
            return True

        except Exception as e:
            print(f"[SigLIP2ONNXTextEncoder] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            return False

    # ------------------------------------------------------------------
    def _extract_text_submodel(self) -> Optional[bytes]:
        """
        Extract the text-encoder-only subgraph from the combined model.

        For SigLIP2: ``{input_ids, attention_mask} -> text_embedding``.
        """
        try:
            import onnx
            from onnx.utils import Extractor
        except ImportError:
            print("[SigLIP2ONNXTextEncoder] 'onnx' package not installed; "
                  "cannot extract text subgraph, will use full model")
            return None

        try:
            print("[SigLIP2ONNXTextEncoder] Extracting text-only subgraph "
                  "from combined model ...")
            model = onnx.load(self.onnx_model_path)

            all_inputs = [inp.name for inp in model.graph.input]
            text_inputs = [n for n in all_inputs if n != "image"]
            if not text_inputs:
                print("[SigLIP2ONNXTextEncoder] Could not determine text "
                      "inputs; falling back to full model")
                return None

            extractor = Extractor(model)
            text_model = extractor.extract_model(
                input_names=text_inputs,
                output_names=["text_embedding"],
            )
            model_bytes = text_model.SerializeToString()
            size_mb = len(model_bytes) / (1024 * 1024)
            print(f"[SigLIP2ONNXTextEncoder] Text-only subgraph extracted "
                  f"({size_mb:.1f} MB, inputs={text_inputs}, "
                  f"image encoder removed)")
            del model, text_model
            return model_bytes
        except Exception as e:
            print(f"[SigLIP2ONNXTextEncoder] Could not extract text "
                  f"subgraph ({e}); falling back to full model")
            return None

    # ------------------------------------------------------------------
    def _find_file(self, suffix: str) -> Optional[str]:
        """Return the first file in *tokenizer_dir* whose name ends with *suffix*."""
        import os
        if not os.path.isdir(self.tokenizer_dir):
            return None
        for fname in os.listdir(self.tokenizer_dir):
            if fname.endswith(suffix):
                return os.path.join(self.tokenizer_dir, fname)
        return None

    def _determine_embedding_dim(self):
        for out in self.session.get_outputs():
            if out.name == "text_embedding":
                shape = out.shape
                if len(shape) >= 2 and isinstance(shape[-1], int):
                    self.embedding_dim = shape[-1]
                    print(f"[SigLIP2ONNXTextEncoder] Detected embedding "
                          f"dim from ONNX output: {self.embedding_dim}")
                return

    def _warmup(self, num_warmup: int = 2):
        print(f"[SigLIP2ONNXTextEncoder] Warming up "
              f"({num_warmup} iterations)...")
        try:
            for _ in range(num_warmup):
                _ = self.encode_texts(["warmup text"])
            print("[SigLIP2ONNXTextEncoder] Warmup complete")
        except Exception as e:
            print(f"[SigLIP2ONNXTextEncoder] Warmup failed "
                  f"(non-critical): {e}")

    # ------------------------------------------------------------------
    @property
    def is_initialized(self) -> bool:
        return (self._initialized
                and self.session is not None
                and self.tokenizer is not None)

    def encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.is_initialized:
            print("[SigLIP2ONNXTextEncoder] Model not initialized")
            return None
        if not texts:
            return np.array([], dtype=np.float32)

        with self._lock:
            try:
                encoded = self.tokenizer(
                    texts,
                    padding="max_length",
                    max_length=self.MAX_LENGTH,
                    truncation=True,
                    return_tensors="np",
                    return_attention_mask=True,
                )
                input_ids = encoded["input_ids"].astype(np.int64)
                attention_mask = encoded["attention_mask"].astype(np.int64)

                if self._text_only:
                    feed = {"input_ids": input_ids}
                    if self._needs_attention_mask:
                        feed["attention_mask"] = attention_mask
                    outputs = self.session.run(
                        ["text_embedding"], feed)
                else:
                    batch_size = input_ids.shape[0]
                    dummy_image = np.zeros(
                        (batch_size, 3, self._image_size, self._image_size),
                        dtype=np.float32,
                    )
                    feed = {
                        "image": dummy_image,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }
                    outputs = self.session.run(
                        ["text_embedding"], feed)

                text_emb = outputs[0].astype(np.float32)

                norms = np.linalg.norm(text_emb, axis=-1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                text_emb = text_emb / norms

                return text_emb

            except Exception as e:
                print(f"[SigLIP2ONNXTextEncoder] Error encoding texts: {e}")
                import traceback
                traceback.print_exc()
                return None

    def encode_single(self, text: str) -> Optional[np.ndarray]:
        result = self.encode_texts([text])
        if result is not None and len(result) > 0:
            return result[0]
        return None


class NvDsTextEmbedder(GstBase.BaseTransform):
    """
    NvDsTextEmbedder GStreamer Plugin
    
    A custom GStreamer element that generates text embeddings.
    Supports original DFN5B-CLIP (1024-dim) or SigLIP2 Giant (1536-dim) text encoder.
    Selection via model-name property (set from curl/API, e.g. onnx model name).
    
    The plugin operates as a passthrough element for video data, while providing
    text embedding functionality via properties. Text embeddings are computed
    when the 'texts' property is set or when the pipeline starts.
    
    For embedding queries (e.g. from curl/API): ensure the pipeline is in PLAYING
    state before sending the first query so the model is loaded; otherwise the
    first request may time out (120s) or return an error.
    
    Properties:
        texts: JSON array of texts to encode
        embeddings: JSON array of computed embeddings (read-only)
        batch-size: Maximum batch size for inference
        verbose: Enable verbose logging
        device: Device for inference (cuda/cpu)
        use-fp16: Enable FP16 inference
        model-name: Set from curl/API (e.g. onnx model name). Contains 'siglip2-giant' or 'giant' -> SigLIP2 Giant; else DFN5B-CLIP
        embedding-dim: Model embedding dimension (read-only)
    """
    
    # GStreamer plugin metadata
    __gstmetadata__ = (
        "NvDsTextEmbedder",                     # Long name
        "Transform/Text/AI",                     # Classification
        "DeepStream Text Embedder Plugin - Generates text embeddings (SigLIP 2 Giant default, DFN5B-CLIP optional)",
        "DS_Copilot"                            # Author
    )
    
    # Pad templates
    __gsttemplates__ = (SINK_TEMPLATE, SRC_TEMPLATE)
    
    # Plugin properties
    __gproperties__ = {
        "texts": (
            GObject.TYPE_STRING,
            "Texts",
            "JSON array of texts to encode (e.g., '[\"query1\", \"query2\"]')",
            "[]",
            GObject.ParamFlags.READWRITE
        ),
        "embeddings": (
            GObject.TYPE_STRING,
            "Embeddings",
            "JSON array of computed embeddings (read-only)",
            "[]",
            GObject.ParamFlags.READABLE
        ),
        "batch-size": (
            GObject.TYPE_INT,
            "Batch Size",
            "Maximum batch size for inference",
            1,                                  # Minimum
            128,                                # Maximum
            64,                                 # Default
            GObject.ParamFlags.READWRITE
        ),
        "verbose": (
            GObject.TYPE_BOOLEAN,
            "Verbose",
            "Enable verbose logging",
            False,
            GObject.ParamFlags.READWRITE
        ),
        "device": (
            GObject.TYPE_STRING,
            "Device",
            "Device for inference (cuda or cpu)",
            "cuda",
            GObject.ParamFlags.READWRITE
        ),
        "use-fp16": (
            GObject.TYPE_BOOLEAN,
            "Use FP16",
            "Enable FP16 inference for faster processing",
            True,
            GObject.ParamFlags.READWRITE
        ),
        "model-name": (
            GObject.TYPE_STRING,
            "Model Name",
            "Encoder selection: set from curl/API (e.g. onnx model name). Use name containing 'siglip2-giant' or 'giant' for SigLIP2 Giant (1536-dim); use 'dfn5b-clip' for DFN5B-CLIP (1024-dim)",
            "siglip2-giant",
            GObject.ParamFlags.READWRITE
        ),
        "embedding-dim": (
            GObject.TYPE_INT,
            "Embedding Dimension",
            "Model embedding dimension (read-only)",
            0,
            65536,
            1536,
            GObject.ParamFlags.READABLE
        ),
        "num-texts": (
            GObject.TYPE_INT,
            "Number of Texts",
            "Number of texts currently loaded (read-only)",
            0,
            GObject.G_MAXINT,
            0,
            GObject.ParamFlags.READABLE
        ),
        "model-loaded": (
            GObject.TYPE_BOOLEAN,
            "Model Loaded",
            "Whether the model is loaded and ready (read-only)",
            False,
            GObject.ParamFlags.READABLE
        ),
        "last-query-result": (
            GObject.TYPE_STRING,
            "Last Query Result",
            "JSON string of last embedding query result (read-only)",
            "{}",
            GObject.ParamFlags.READABLE
        ),
        "onnx-model-path": (
            GObject.TYPE_STRING,
            "ONNX Model Path",
            "Path to the DFN-CLIP ONNX model file (used when model-name contains 'dfnclip')",
            "",
            GObject.ParamFlags.READWRITE
        ),
        "tokenizer-dir": (
            GObject.TYPE_STRING,
            "Tokenizer Directory",
            "Path to the tokenizer directory with tokenizer.json (used when model-name contains 'dfnclip')",
            "",
            GObject.ParamFlags.READWRITE
        ),
    }
    
    def __init__(self):
        """Initialize the plugin instance."""
        super(NvDsTextEmbedder, self).__init__()
        
        # Initialize properties
        self._texts_json = "[]"
        self._texts = []
        self._embeddings = []
        self._embeddings_json = "[]"
        self.batch_size = 64
        self.verbose = False
        self.device = "cuda"
        self.use_fp16 = True
        self._model_name = "siglip2-giant"
        self._onnx_model_path = ""
        self._tokenizer_dir = ""
        
        # Initialize encoder (lazy loading); type: DFN5BCLIPEncoder, SigLIP2GiantTextEncoder, or DFNClipONNXTextEncoder
        self.encoder = None
        self._lazy_init_pending = False
        
        # Thread lock for protecting encoder initialization and queries
        # Critical for C GStreamer pipeline integration
        self._encoder_lock = threading.Lock()
        
        # Encoding thread and queues for async processing
        # Input queue: (request_id, texts) tuples
        # Output queue: (request_id, embeddings, error) tuples
        self._input_queue: queue.Queue = queue.Queue()
        self._output_queue: queue.Queue = queue.Queue()
        self._pending_requests: Dict[str, threading.Event] = {}
        self._pending_results: Dict[str, Tuple[Optional[np.ndarray], Optional[str]]] = {}
        self._pending_lock = threading.Lock()
        self._encoder_thread: Optional[threading.Thread] = None
        self._stop_encoder_thread = threading.Event()
        
        # Last query result storage (for Python GStreamer bindings workaround)
        # Since writable_structure() doesn't work in Python, we store results here
        self._last_query_result = {
            'success': False,
            'embeddings': [],
            'num_embeddings': 0,
            'embedding_dim': 0,
            'error': None
        }
        
        # Set passthrough mode (we don't modify video data)
        self.set_passthrough(True)
        
        # Set in-place mode
        self.set_in_place(True)
        
        Gst.info("NvDsTextEmbedder: Plugin initialized")
    
    def do_get_property(self, prop):
        """Handle property get requests."""
        if prop.name == 'texts':
            return self._texts_json
        elif prop.name == 'embeddings':
            return self._embeddings_json
        elif prop.name == 'batch-size':
            return self.batch_size
        elif prop.name == 'verbose':
            return self.verbose
        elif prop.name == 'device':
            return self.device
        elif prop.name == 'use-fp16':
            return self.use_fp16
        elif prop.name == 'model-name':
            return self._model_name
        elif prop.name == 'onnx-model-path':
            return self._onnx_model_path
        elif prop.name == 'tokenizer-dir':
            return self._tokenizer_dir
        elif prop.name == 'embedding-dim':
            if self.encoder and self.encoder.is_initialized:
                return self.encoder.embedding_dim
            if self._use_dfnclip_onnx_model():
                return DFNClipONNXTextEncoder.EMBEDDING_DIM
            if self._use_siglip2_onnx_model():
                return SigLIP2ONNXTextEncoder.EMBEDDING_DIM
            if self._use_siglip2_giant_model():
                return SigLIP2GiantTextEncoder.EMBEDDING_DIM
            return DFN5BCLIPEncoder.EMBEDDING_DIM
        elif prop.name == 'num-texts':
            return len(self._texts)
        elif prop.name == 'model-loaded':
            return self.encoder is not None and self.encoder.is_initialized
        elif prop.name == 'last-query-result':
            return json.dumps(self._last_query_result)
        else:
            raise AttributeError(f'Unknown property: {prop.name}')
    
    def do_set_property(self, prop, value):
        """Handle property set requests."""
        if prop.name == 'texts':
            self._set_texts(value)
        elif prop.name == 'batch-size':
            self.batch_size = value
        elif prop.name == 'verbose':
            self.verbose = value
        elif prop.name == 'device':
            self.device = value
        elif prop.name == 'use-fp16':
            self.use_fp16 = value
        elif prop.name == 'model-name':
            self._model_name = (value or "siglip2-giant").strip()
        elif prop.name == 'onnx-model-path':
            self._onnx_model_path = (value or "").strip()
        elif prop.name == 'tokenizer-dir':
            self._tokenizer_dir = (value or "").strip()
        else:
            raise AttributeError(f'Unknown property or read-only: {prop.name}')
    
    def _use_dfnclip_onnx_model(self) -> bool:
        """True if model-name selects DFN-CLIP ONNX text encoder."""
        name = (self._model_name or "").lower()
        return "dfnclip" in name or "dfn-clip" in name or "dfn_clip" in name or "radio-clip" in name or "radio_clip" in name

    def _use_siglip2_onnx_model(self) -> bool:
        """True if model-name selects SigLIP2 ONNX text encoder."""
        name = (self._model_name or "").lower()
        return "siglip2-onnx" in name or "siglip2_onnx" in name

    def _use_siglip2_giant_model(self) -> bool:
        """True if model-name selects SigLIP2 Giant PyTorch text encoder."""
        if self._use_dfnclip_onnx_model() or self._use_siglip2_onnx_model():
            return False
        name = (self._model_name or "").lower()
        return "siglip2-giant" in name or "giant" in name

    def _set_texts(self, texts_json: str):
        """
        Set texts from JSON string and compute embeddings if model is ready.
        
        Args:
            texts_json: JSON array string of texts (e.g., '["query1", "query2"]')
        """
        try:
            # Parse JSON
            texts = json.loads(texts_json)
            
            if not isinstance(texts, list):
                print(f"[NvDsTextEmbedder] Error: texts must be a JSON array, got {type(texts)}")
                return
            
            # Validate all items are strings
            texts = [str(t) for t in texts]
            
            self._texts_json = texts_json
            self._texts = texts
            
            if self.verbose:
                print(f"[NvDsTextEmbedder] Set {len(texts)} texts")
            
            # If encoder is ready, compute embeddings immediately
            if self.encoder and self.encoder.is_initialized:
                self._compute_embeddings()
            
        except json.JSONDecodeError as e:
            print(f"[NvDsTextEmbedder] Error parsing texts JSON: {e}")
        except Exception as e:
            print(f"[NvDsTextEmbedder] Error setting texts: {e}")
    
    def _compute_embeddings(self):
        """
        Compute embeddings for the current texts.
        
        Thread-safe: Uses the encoder's internal lock for inference operations.
        """
        if not self._texts:
            self._embeddings = []
            self._embeddings_json = "[]"
            return
        
        if not self.encoder or not self.encoder.is_initialized:
            print("[NvDsTextEmbedder] Cannot compute embeddings: model not initialized")
            return
        
        try:
            if self.verbose:
                print(f"[NvDsTextEmbedder] Computing embeddings for {len(self._texts)} texts...")
            
            # Process in batches (encode_texts is thread-safe)
            all_embeddings = []
            
            for i in range(0, len(self._texts), self.batch_size):
                batch = self._texts[i:i + self.batch_size]
                embeddings = self.encoder.encode_texts(batch)
                
                if embeddings is not None:
                    all_embeddings.extend(embeddings.tolist())
            
            self._embeddings = all_embeddings
            self._embeddings_json = json.dumps(all_embeddings)
            
            if self.verbose:
                print(f"[NvDsTextEmbedder] Computed {len(all_embeddings)} embeddings")
            
        except Exception as e:
            print(f"[NvDsTextEmbedder] Error computing embeddings: {e}")
            import traceback
            traceback.print_exc()
    
    def do_start(self):
        """
        Called when the element starts processing.
        
        Uses lazy initialization - model loading is deferred to prevent
        CUDA operations from interfering with GStreamer pipeline setup.
        
        Returns:
            bool: True on success, False on failure
        """
        Gst.info("NvDsTextEmbedder: Starting (lazy init enabled)...")
        
        # Defer model loading until needed
        self._lazy_init_pending = True
        self.encoder = None
        
        # Reset thread control
        self._stop_encoder_thread.clear()
        
        # Start encoder thread
        self._encoder_thread = threading.Thread(
            target=self._encoder_thread_worker,
            name="NvDsTextEmbedder-EncoderThread",
            daemon=True
        )
        self._encoder_thread.start()
        print("[NvDsTextEmbedder] Encoder thread launched")
        
        # Set custom query function on src pad to handle embedding queries
        # This allows other pipeline components to send queries to this element
        src_pad = self.get_static_pad("src")
        if src_pad:
            src_pad.set_query_function(self._src_pad_query_function)
        
        # Also set on sink pad for upstream queries
        sink_pad = self.get_static_pad("sink")
        if sink_pad:
            sink_pad.set_query_function(self._sink_pad_query_function)
        
        return True
    
    def _src_pad_query_function(self, pad: Gst.Pad, parent: Gst.Object, query: Gst.Query) -> bool:
        """
        Custom query function for src pad.
        
        This handles custom "embedding" queries and chains to default for others.
        Other pipeline components can send queries to this element using:
            pad.query(query) or element.query(query)
        
        Args:
            pad: The src pad
            parent: Parent element (self)
            query: The query to handle
            
        Returns:
            True if query was handled, False otherwise
        """
        # Handle custom embedding queries
        if query.type == Gst.QueryType.CUSTOM:
            structure = query.get_structure()
            if structure and structure.get_name() == EMBEDDING_QUERY_NAME:
                return self._handle_custom_query(query)
        
        # Chain to default query function for other queries
        return pad.query_default(parent, query)
    
    def _sink_pad_query_function(self, pad: Gst.Pad, parent: Gst.Object, query: Gst.Query) -> bool:
        """
        Custom query function for sink pad.
        
        This handles custom "embedding" queries sent upstream.
        
        Args:
            pad: The sink pad
            parent: Parent element (self)
            query: The query to handle
            
        Returns:
            True if query was handled, False otherwise
        """
        # Handle custom embedding queries
        if query.type == Gst.QueryType.CUSTOM:
            structure = query.get_structure()
            if structure and structure.get_name() == EMBEDDING_QUERY_NAME:
                return self._handle_custom_query(query)
        
        # Chain to default query function for other queries
        return pad.query_default(parent, query)
    
    def _lazy_init_encoder(self):
        """
        Lazy initialization of the encoder.
        Called on first use to ensure GStreamer pipeline is fully set up.
        
        Thread-safe: Uses lock to prevent concurrent initialization from
        multiple threads (common in C GStreamer pipelines).
        """
        # Double-checked locking pattern for thread safety
        if not self._lazy_init_pending:
            return
        
        with self._encoder_lock:
            # Re-check after acquiring lock
            if not self._lazy_init_pending:
                return
            
            self._lazy_init_pending = False
            print("[NvDsTextEmbedder] Performing lazy initialization of encoder...")
            
            if self._use_dfnclip_onnx_model():
                # DFN-CLIP via ONNX Runtime
                if not self._onnx_model_path or not self._tokenizer_dir:
                    Gst.warning("NvDsTextEmbedder: dfnclip selected but "
                                "onnx-model-path and/or tokenizer-dir not set "
                                "in config file")
                    self.encoder = None
                elif ONNXRUNTIME_AVAILABLE and TRANSFORMERS_AVAILABLE:
                    self.encoder = DFNClipONNXTextEncoder(
                        onnx_model_path=self._onnx_model_path,
                        tokenizer_dir=self._tokenizer_dir,
                        device=self.device,
                    )
                    if not self.encoder.load_model():
                        Gst.warning("NvDsTextEmbedder: Failed to load DFN-CLIP ONNX encoder")
                        self.encoder = None
                    else:
                        print(f"[NvDsTextEmbedder] Loaded DFN-CLIP ONNX text encoder "
                              f"(embedding_dim={self.encoder.embedding_dim})")
                        if self._texts:
                            self._compute_embeddings()
                else:
                    missing = []
                    if not ONNXRUNTIME_AVAILABLE:
                        missing.append("onnxruntime")
                    if not TRANSFORMERS_AVAILABLE:
                        missing.append("transformers")
                    Gst.warning(f"NvDsTextEmbedder: Missing packages for DFN-CLIP ONNX: "
                                f"{', '.join(missing)}")
                    self.encoder = None
            elif self._use_siglip2_onnx_model():
                # SigLIP2 via ONNX Runtime
                if not self._onnx_model_path or not self._tokenizer_dir:
                    Gst.warning("NvDsTextEmbedder: siglip2-onnx selected but "
                                "onnx-model-path and/or tokenizer-dir not set "
                                "in config file")
                    self.encoder = None
                elif ONNXRUNTIME_AVAILABLE and TRANSFORMERS_AVAILABLE:
                    self.encoder = SigLIP2ONNXTextEncoder(
                        onnx_model_path=self._onnx_model_path,
                        tokenizer_dir=self._tokenizer_dir,
                        device=self.device,
                    )
                    if not self.encoder.load_model():
                        Gst.warning("NvDsTextEmbedder: Failed to load SigLIP2 ONNX encoder")
                        self.encoder = None
                    else:
                        print(f"[NvDsTextEmbedder] Loaded SigLIP2 ONNX text encoder "
                              f"(embedding_dim={self.encoder.embedding_dim})")
                        if self._texts:
                            self._compute_embeddings()
                else:
                    missing = []
                    if not ONNXRUNTIME_AVAILABLE:
                        missing.append("onnxruntime")
                    if not TRANSFORMERS_AVAILABLE:
                        missing.append("transformers")
                    Gst.warning(f"NvDsTextEmbedder: Missing packages for SigLIP2 ONNX: "
                                f"{', '.join(missing)}")
                    self.encoder = None
            elif self._use_siglip2_giant_model():
                # SigLIP2 Giant text encoder
                if ML_AVAILABLE and TRANSFORMERS_AVAILABLE:
                    self.encoder = SigLIP2GiantTextEncoder(device=self.device, use_fp16=self.use_fp16)
                    if not self.encoder.load_model():
                        Gst.warning("NvDsTextEmbedder: Failed to load SigLIP2 Giant text encoder")
                        self.encoder = None
                    else:
                        print(f"[NvDsTextEmbedder] Loaded SigLIP2 Giant text encoder: {SigLIP2GiantTextEncoder.MODEL_ID} (embedding_dim={self.encoder.embedding_dim})")
                        if self._texts:
                            self._compute_embeddings()
                else:
                    Gst.warning("NvDsTextEmbedder: transformers not available for SigLIP2 Giant")
                    self.encoder = None
            else:
                # Original: DFN5B-CLIP
                if ML_AVAILABLE and OPENCLIP_AVAILABLE:
                    self.encoder = DFN5BCLIPEncoder(device=self.device, use_fp16=self.use_fp16)
                    if not self.encoder.load_model():
                        Gst.warning("NvDsTextEmbedder: Failed to load DFN5B-CLIP model")
                        self.encoder = None
                    else:
                        print(f"[NvDsTextEmbedder] Loaded DFN5B-CLIP text encoder: {DFN5BCLIPEncoder.MODEL_NAME} (embedding_dim={self.encoder.embedding_dim})")
                        if self._texts:
                            self._compute_embeddings()
                else:
                    Gst.warning("NvDsTextEmbedder: ML libraries not available, embeddings disabled")
                    self.encoder = None
    
    def _encoder_thread_worker(self):
        """
        Worker thread that processes encoding requests from the input queue.
        
        Reads (request_id, texts) from input_queue, performs encoding,
        and pushes (request_id, embeddings, error) to output_queue.
        """
        print("[NvDsTextEmbedder] Encoder thread started")
        
        while not self._stop_encoder_thread.is_set():
            try:
                # Wait for input with timeout to allow checking stop flag
                try:
                    request_id, texts = self._input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if self.verbose:
                    print(f"[NvDsTextEmbedder] Encoder thread processing request {request_id} with {len(texts)} texts")
                
                # Ensure encoder is initialized
                if self._lazy_init_pending:
                    self._lazy_init_encoder()
                
                # Perform encoding
                embeddings = None
                error = None
                
                if not self.encoder or not self.encoder.is_initialized:
                    error = "Model not initialized"
                else:
                    try:
                        embeddings = self.encoder.encode_texts(texts)
                        if embeddings is None:
                            error = "Encoding failed"
                    except Exception as e:
                        error = str(e)
                        print(f"[NvDsTextEmbedder] Encoder thread error: {e}")
                # Store result and signal completion
                with self._pending_lock:
                    self._pending_results[request_id] = (embeddings, error)
                    if request_id in self._pending_requests:
                        self._pending_requests[request_id].set()
                
                self._input_queue.task_done()
                
            except Exception as e:
                print(f"[NvDsTextEmbedder] Encoder thread exception: {e}")
                import traceback
                traceback.print_exc()
        
        print("[NvDsTextEmbedder] Encoder thread stopped")
    
    def do_change_state(self, transition: Gst.StateChange) -> Gst.StateChangeReturn:
        """
        Handle state changes.

        Performs model loading during PAUSED->PLAYING transition to ensure
        the model is ready before any buffers flow through the pipeline.

        Args:
            transition: The state transition being performed

        Returns:
            Gst.StateChangeReturn indicating success or failure
        """
        if transition == Gst.StateChange.PAUSED_TO_PLAYING:
            self._lazy_init_encoder()

            # Check if initialization succeeded
            if self.encoder is None or not self.encoder.is_initialized:
                if ML_AVAILABLE and OPENCLIP_AVAILABLE:
                    Gst.warning("NvDsTextEmbedder: Failed to initialize encoder during state change")
                    # Return FAILURE to prevent the state change if ML is expected to work
                    # Comment out the next line if you want to allow pipeline to run without ML
                    # return Gst.StateChangeReturn.FAILURE

        # Chain to parent class
        return GstBase.BaseTransform.do_change_state(self, transition)

    def do_stop(self):
        """
        Called when the element stops processing.
        Clean up resources.
        
        Thread-safe cleanup with proper CUDA synchronization.
        
        Returns:
            bool: True on success, False on failure
        """
        Gst.info(f"NvDsTextEmbedder: Stopping... Texts: {len(self._texts)}")
        
        # Stop encoder thread
        if self._encoder_thread is not None and self._encoder_thread.is_alive():
            print("[NvDsTextEmbedder] Stopping encoder thread...")
            self._stop_encoder_thread.set()
            
            # Signal any waiting requests to unblock
            with self._pending_lock:
                for event in self._pending_requests.values():
                    event.set()
            
            self._encoder_thread.join(timeout=5.0)
            if self._encoder_thread.is_alive():
                print("[NvDsTextEmbedder] Warning: Encoder thread did not stop in time")
            else:
                print("[NvDsTextEmbedder] Encoder thread stopped")
            self._encoder_thread = None
        
        # Clear queues and pending requests
        with self._pending_lock:
            self._pending_requests.clear()
            self._pending_results.clear()
        
        # Drain queues
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break
        
        # Thread-safe cleanup
        with self._encoder_lock:
            # Clean up encoder
            if self.encoder is not None:
                # Synchronize CUDA before cleanup to ensure all operations complete
                if ML_AVAILABLE:
                    try:
                        torch.cuda.synchronize()
                    except:
                        pass
                
                self.encoder.model = None
                if hasattr(self.encoder, "tokenizer"):
                    self.encoder.tokenizer = None
                if hasattr(self.encoder, "processor"):
                    self.encoder.processor = None
                self.encoder = None
            
            # Clear CUDA cache if available
            if ML_AVAILABLE:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
        
        return True
    
    def do_transform_ip(self, buffer: Gst.Buffer) -> Gst.FlowReturn:
        """
        In-place transform function. This is called for each buffer.
        
        For the text embedder, this is essentially a passthrough - we don't
        modify the video data. The text embeddings are computed when the
        'texts' property is set.
        
        Note: Model initialization happens during PAUSED->PLAYING state change,
        so the encoder should already be ready when buffers start flowing.
        
        Args:
            buffer: The input/output buffer (passed through unchanged)
            
        Returns:
            Gst.FlowReturn.OK on success
        """

        return Gst.FlowReturn.OK
    
    def get_embeddings(self) -> List[List[float]]:
        """
        Get the computed embeddings as a Python list.
        
        Returns:
            List of embedding vectors
        """
        return self._embeddings
    
    def get_embeddings_numpy(self) -> Optional[np.ndarray]:
        """
        Get the computed embeddings as a numpy array.
        
        Returns:
            Numpy array of embeddings (N, embedding_dim) or None if no embeddings
        """
        if not self._embeddings:
            return None
        return np.array(self._embeddings, dtype=np.float32)
    
    def encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Directly encode texts without going through properties.
        
        This is a convenience method for Python users who want to
        encode texts directly without JSON serialization.
        
        Thread Safety:
            This method is thread-safe and can be called from C GStreamer pipelines.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of embeddings (N, embedding_dim) or None on error
        """
        # Ensure encoder is initialized (thread-safe)
        if self._lazy_init_pending:
            self._lazy_init_encoder()
        
        if not self.encoder or not self.encoder.is_initialized:
            print("[NvDsTextEmbedder] Model not initialized")
            return None
        
        # encode_texts is thread-safe and includes CUDA synchronization
        return self.encoder.encode_texts(texts)
    
    def _handle_custom_query(self, query: Gst.Query) -> bool:
        """
        Handle custom embedding query.
        
        Thread Safety:
            This method is thread-safe and can be called from C GStreamer pipelines.
            Encoding is performed in a separate thread. This method pushes the request
            to the input queue and waits for the result from the output queue.
        
        Args:
            query: The custom GStreamer query
            
        Returns:
            True if query was handled successfully, False otherwise
        """
        try:
            # Get the query structure
            structure = query.get_structure()
            
            if structure is None:
                return False
            
            # Check if this is an embedding query
            struct_name = structure.get_name()
            if struct_name != EMBEDDING_QUERY_NAME:
                # Not our query, let parent handle it
                return False
            
            if self.verbose:
                print(f"[NvDsTextEmbedder] Received embedding query")
            
            # Get the text from the query structure
            # Note: In newer GStreamer, get_string returns string directly, not tuple
            if not structure.has_field("text"):
                print("[NvDsTextEmbedder] Embedding query missing 'text' field")
                self._set_query_result(query, False, None, "Missing 'text' field")
                return True
            
            text_value = structure.get_string("text")
            # PyGObject can return (value, success) tuple in some versions
            if isinstance(text_value, (list, tuple)) and len(text_value) >= 1:
                text_value = text_value[0]
            if not text_value:
                print("[NvDsTextEmbedder] Embedding query has empty 'text' field")
                self._set_query_result(query, False, None, "Empty 'text' field")
                return True
            
            # Parse the text (can be a single string or JSON array)
            try:
                # Try parsing as JSON array first
                texts = json.loads(text_value)
                if isinstance(texts, str):
                    texts = [texts]
                elif not isinstance(texts, list):
                    texts = [str(texts)]
            except json.JSONDecodeError:
                # Treat as single text string
                texts = [text_value]
            
            if self.verbose:
                print(f"[NvDsTextEmbedder] Encoding {len(texts)} texts from query via encoder thread")
            
            # Check if encoder thread is running
            if self._encoder_thread is None or not self._encoder_thread.is_alive():
                print("[NvDsTextEmbedder] Encoder thread not running")
                self._set_query_result(query, False, None, "Encoder thread not running")
                return True
            
            # Generate unique request ID
            request_id = str(uuid.uuid4())
            
            # Create event for waiting on result
            result_event = threading.Event()
            
            # Register pending request
            with self._pending_lock:
                self._pending_requests[request_id] = result_event
            
            # Push request to input queue
            self._input_queue.put((request_id, texts))
            
            if self.verbose:
                print(f"[NvDsTextEmbedder] Pushed request {request_id} to input queue")
            
            # Wait for result from encoder thread (timeout so curl/client gets a response)
            EMBEDDING_QUERY_TIMEOUT_SEC = 120
            if not result_event.wait(timeout=EMBEDDING_QUERY_TIMEOUT_SEC):
                with self._pending_lock:
                    self._pending_requests.pop(request_id, None)
                self._set_query_result(
                    query, False, None,
                    "Encoding timed out (model may still be loading or overloaded)",
                    request_id=request_id,
                )
                return True
            
            # Retrieve result
            with self._pending_lock:
                embeddings, error = self._pending_results.pop(request_id, (None, "Result not found"))
                self._pending_requests.pop(request_id, None)
            
            if self.verbose:
                print(f"[NvDsTextEmbedder] Received result for request {request_id}")
            
            # Check for errors
            if error is not None:
                self._set_query_result(query, False, None, error, request_id=request_id)
                return True
            
            if embeddings is None:
                self._set_query_result(query, False, None, "Encoding failed", request_id=request_id)
                return True
            
            # Convert to list - embeddings is already a copied numpy array
            # with fully synchronized CUDA operations
            embeddings_list = embeddings.tolist()
            # Set the result in the query structure
            self._set_query_result(query, True, embeddings_list, request_id=request_id)
            
            if self.verbose:
                print(f"[NvDsTextEmbedder] Query returned {len(embeddings_list)} embeddings")
            
            return True
            
        except Exception as e:
            print(f"[NvDsTextEmbedder] Error handling embedding query: {e}")
            import traceback
            traceback.print_exc()
            # Return True to indicate we handled the query (with error)
            # This prevents the query from being passed to other handlers
            self._set_query_result(query, False, None, str(e))
            return True
    
    def _set_query_result(self, query: Gst.Query, success: bool, 
                          embeddings: Optional[List[List[float]]], 
                          error_msg: str = None,
                          request_id: str = None):
        """
        Set the query result directly in the query structure.
        
        Uses ctypes to bypass Python GStreamer bindings limitation where 
        writable_structure() returns a copy. This ensures results are returned
        synchronously in the query structure as expected by callers.
        
        Sets fields compatible with gst_nvquery_text_embedding_parse():
            - id: request identifier string
            - created: timestamp (uint64)
            - response-model: model name string
            - data: JSON string containing embeddings array
        
        Args:
            query: The GStreamer query to store results in
            success: Whether the encoding succeeded
            embeddings: List of embedding vectors (or None on failure)
            error_msg: Error message if failed
            request_id: Optional request ID for tracking
        """
        import time
        
        # Set values directly on the query structure using ctypes
        _set_query_structure_value(query, "success", success)
        
        # Set fields expected by gst_nvquery_text_embedding_parse()
        # id - request identifier
        if request_id:
            _set_query_structure_value(query, "id", request_id)
        else:
            _set_query_structure_value(query, "id", str(uuid.uuid4()))
        
        # created - timestamp in nanoseconds (uint64)
        created_ns = int(time.time() * 1_000_000_000)
        _set_query_structure_value(query, "created", created_ns, value_type='uint64')
        
        
        if embeddings is not None:
            _set_query_structure_value(query, "embeddings", json.dumps(embeddings))
            _set_query_structure_value(query, "num_embeddings", len(embeddings))
            if embeddings:
                _set_query_structure_value(query, "embedding_dim", len(embeddings[0]))
            else:
                _set_query_structure_value(query, "embedding_dim", 0)
        else:
            _set_query_structure_value(query, "embeddings", "[]")
            _set_query_structure_value(query, "num_embeddings", 0)
            _set_query_structure_value(query, "embedding_dim", 0)
        
        if error_msg:
            _set_query_structure_value(query, "error", error_msg)
        
        # Also store in instance variable for backward compatibility
        self._last_query_result = {
            'success': success,
            'embeddings': embeddings if embeddings is not None else [],
            'num_embeddings': len(embeddings) if embeddings else 0,
            'embedding_dim': len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0,
            'error': error_msg
        }


def create_embedding_query(texts: List[str]) -> Gst.Query:
    """
    Create a custom embedding query to send to nvdstextembedder element.
    
    This helper function creates a properly formatted query that can be
    sent to the element to generate text embeddings on demand.
    
    Args:
        texts: List of text strings to encode
        
    Returns:
        Gst.Query object ready to be sent to the element
        
    Example:
        >>> query = create_embedding_query(["person in red", "blue car"])
        >>> element.query(query)
        >>> result = parse_embedding_query_result(query)
        >>> print(result['embeddings'])
    """
    # Create structure with the text to encode
    structure = Gst.Structure.new_empty(EMBEDDING_QUERY_NAME)
    structure.set_value("text", json.dumps(texts))
    
    # Create custom query with the structure
    query = Gst.Query.new_custom(Gst.QueryType.CUSTOM, structure)
    
    return query


def create_embedding_query_single(text: str) -> Gst.Query:
    """
    Create a custom embedding query for a single text.
    
    Args:
        text: Single text string to encode
        
    Returns:
        Gst.Query object ready to be sent to the element
    """
    return create_embedding_query([text])


def get_embedding_query_result(element: Gst.Element) -> Dict[str, Any]:
    """
    Get the result from the last embedding query from the element.
    
    Due to Python GStreamer bindings limitations, query results are stored
    in the element's 'last-query-result' property instead of the query structure.
    Call this function after sending a query to retrieve the results.
    
    Args:
        element: The nvdstextembedder element that processed the query
        
    Returns:
        Dictionary containing:
            - success: bool - whether encoding succeeded
            - embeddings: List[List[float]] - the embedding vectors
            - num_embeddings: int - number of embeddings
            - embedding_dim: int - dimension of each embedding
            - error: str - error message if failed (optional)
            
    Example:
        >>> query = create_embedding_query(["hello world"])
        >>> pad.query(query)  # Returns True if query was handled
        >>> result = get_embedding_query_result(embedder_element)
        >>> if result['success']:
        ...     embeddings = np.array(result['embeddings'])
    """
    try:
        result_json = element.get_property("last-query-result")
        return json.loads(result_json)
    except Exception as e:
        return {
            'success': False,
            'embeddings': [],
            'num_embeddings': 0,
            'embedding_dim': 0,
            'error': str(e)
        }


def parse_embedding_query_result(query: Gst.Query) -> Dict[str, Any]:
    """
    Parse the result from an embedding query structure.
    
    This function reads the results that were set on the query structure
    by the nvdstextembedder element. Results are returned synchronously
    in the query structure.
    
    Args:
        query: The query after it has been processed by the element
        
    Returns:
        Dictionary containing:
            - success: bool - whether encoding succeeded
            - embeddings: List[List[float]] - the embedding vectors
            - num_embeddings: int - number of embeddings
            - embedding_dim: int - dimension of each embedding
            - error: str - error message if failed (optional)
            
    Example:
        >>> query = create_embedding_query(["hello world"])
        >>> pad.query(query)
        >>> result = parse_embedding_query_result(query)
        >>> if result['success']:
        ...     embeddings = np.array(result['embeddings'])
    """
    result = {
        'success': False,
        'embeddings': [],
        'num_embeddings': 0,
        'embedding_dim': 0,
        'error': None
    }
    
    try:
        structure = query.get_structure()
        
        if structure is None:
            result['error'] = "No structure in query"
            return result
        
        # Get success flag (get_boolean returns (found, value) tuple in some bindings)
        if structure.has_field("success"):
            success_result = structure.get_boolean("success")
            if isinstance(success_result, (list, tuple)) and len(success_result) >= 2:
                result['success'] = bool(success_result[1])
            elif isinstance(success_result, bool):
                result['success'] = success_result
        
        # Get embeddings JSON (get_string may return (value, found) tuple)
        if structure.has_field("embeddings"):
            embeddings_json = structure.get_string("embeddings")
            if isinstance(embeddings_json, (list, tuple)) and len(embeddings_json) >= 1:
                embeddings_json = embeddings_json[0]
            if embeddings_json:
                try:
                    result['embeddings'] = json.loads(embeddings_json)
                except json.JSONDecodeError:
                    result['embeddings'] = []
        
        # Get num_embeddings (get_int returns (found, value) tuple in some bindings)
        if structure.has_field("num_embeddings"):
            num_result = structure.get_int("num_embeddings")
            if isinstance(num_result, (list, tuple)) and len(num_result) >= 2 and num_result[0]:
                result['num_embeddings'] = int(num_result[1])
        
        # Get embedding_dim
        if structure.has_field("embedding_dim"):
            dim_result = structure.get_int("embedding_dim")
            if isinstance(dim_result, (list, tuple)) and len(dim_result) >= 2 and dim_result[0]:
                result['embedding_dim'] = int(dim_result[1])
        
        # Get error if any (get_string may return tuple)
        if structure.has_field("error"):
            err_val = structure.get_string("error")
            if isinstance(err_val, (list, tuple)) and len(err_val) >= 1:
                err_val = err_val[0]
            result['error'] = err_val
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


# Plugin registration
GObject.type_register(NvDsTextEmbedder)

__gstelementfactory__ = (
    "nvdstextembedder",         # Element name (used in gst-launch)
    Gst.Rank.NONE,              # Rank (priority for autoplugging)
    NvDsTextEmbedder            # Element class
)
