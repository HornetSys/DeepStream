/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file gstnvdsvisionencoder.h
 * @brief GStreamer plugin for vision encoding (ReID embeddings)
 * 
 * This plugin generates ReID embeddings for tracked objects using Triton Inference Server
 * and attaches them as NVDS_TRACKER_OBJ_REID_META to object metadata.
 */

#ifndef __GST_NVDSVISIONENCODER_H__
#define __GST_NVDSVISIONENCODER_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include "nvds_tracker_meta.h"
#include "nvds_opticalflow_meta.h"

#ifdef ENABLE_INSTRUMENTATION
#include <sys/time.h>
#endif

#define DEFAULT_CACHE_REFRESH_INTERVAL 0
#define DEFAULT_CACHE_MAX_SIZE 1000

/* OFA-based embedding prediction defaults */
#define DEFAULT_OFA_MOTION_THRESHOLD 8.0f
#define DEFAULT_OFA_HIGH_MOTION_THRESHOLD 25.0f
#define DEFAULT_OFA_PREDICT_ENABLED FALSE
#define OFA_MOTION_FEATURE_DIM 6

/* DLA two-tier inference defaults */
#define DEFAULT_DLA_VALIDATOR_ENABLED FALSE
#define DEFAULT_DLA_CORE -1
#define DEFAULT_DLA_CHANGE_THRESHOLD 0.3f

/**
 * Cached embedding entry for smart inference.
 * Keyed by tracker-assigned object_id (guint64).
 */
typedef struct {
  float *embedding;
  guint embedding_dim;
  guint64 last_inferred_frame;

  /* Per-object motion history for OFA prediction */
  float last_motion_magnitude;
  float last_flow_dx;
  float last_flow_dy;
  guint64 accumulated_skip_frames;
} CachedEmbedding;

/**
 * Lightweight appearance change detector state.
 * Runs on DLA (Jetson) or GPU fallback (x86) to gate full inference.
 */
typedef struct {
  void *trt_engine;       /* TensorRT engine handle (ICudaEngine*) */
  void *trt_context;      /* TensorRT execution context (IExecutionContext*) */
  float *d_input;         /* Device input buffer for validator */
  float *d_output;        /* Device output: change probability [0,1] */
  guint input_w;
  guint input_h;
  gboolean is_dla;        /* TRUE if running on DLA, FALSE for GPU fallback */
  gint dla_core;
} DlaValidatorCtx;

G_BEGIN_DECLS

#define GST_TYPE_NVDSVISIONENCODER (gst_nvdsvisionencoder_get_type())
#define GST_NVDSVISIONENCODER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSVISIONENCODER,GstNvDsVisionEncoder))
#define GST_NVDSVISIONENCODER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSVISIONENCODER,GstNvDsVisionEncoderClass))
#define GST_IS_NVDSVISIONENCODER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSVISIONENCODER))
#define GST_IS_NVDSVISIONENCODER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSVISIONENCODER))

typedef struct _GstNvDsVisionEncoder GstNvDsVisionEncoder;
typedef struct _GstNvDsVisionEncoderClass GstNvDsVisionEncoderClass;

/**
 * Encoder backend type
 */
typedef enum {
  NVDS_VISION_ENCODER_BACKEND_TRITON = 0,
  NVDS_VISION_ENCODER_BACKEND_TENSORRT = 1,
} NvDsVisionEncoderBackend;

/* Runtime model update (OTA), as gst-nvinfer does it: setting the property
 * queues the path, a loader thread builds a second handle while the live one
 * serves, and the streaming thread swaps them.
 * Prebuilt engine only, and same family only -- crossing families invalidates
 * every embedding already emitted, so it is refused. */
typedef enum {
  NVDS_VISION_ENCODER_LOAD_STOP = 0,     // sentinel: stop the loader thread
  NVDS_VISION_ENCODER_LOAD_FROM_ENGINE,  // tensorrt-engine=<path.plan>
} NvDsVisionEncoderModelLoadType;

/* A queued request: a full snapshot, so the loader never reads element config
 * a property setter could free underneath it. */
typedef struct {
  NvDsVisionEncoderModelLoadType type;
  gchar   *path;          // the engine file being loaded
  gchar   *triton_url;
  gchar   *triton_model;
  guint    batch_size;
  guint    gpu_id;
} NvDsVisionEncoderLoadRequest;

/* A built, validated encoder awaiting commit. Owns @client until it goes live
 * or is discarded. */
typedef struct {
  void    *client;
  guint    input_width;
  guint    input_height;
  guint    embedding_dim;
  gchar   *path;
  guint    max_batch_size;  // engine capacity, validated before commit
  guint    gpu_id;          // device this client's memory belongs to
  NvDsVisionEncoderModelLoadType type;
} NvDsVisionEncoderReplacement;

/**
 * GstNvDsVisionEncoder structure
 */
struct _GstNvDsVisionEncoder {
  GstBaseTransform base_trans;
  
  /* Configuration properties */
  gboolean enable;
  NvDsVisionEncoderBackend backend;
  gchar *backend_str;  // String version for property
  gchar *triton_url;
  gchar *triton_model;
  gchar *model_variant;  // For compatibility with Python plugin config
  gchar *tensorrt_engine;
  gchar *onnx_model;     // Path to ONNX model for engine building
  guint batch_size;      // TensorRT batch size (for engine building/selection)
  guint min_crop_size;
  gboolean verbose;
  guint gpu_id;
  gchar *device;  // For compatibility (cuda/cpu)
  guint skip_interval;  // Frame skip interval: 0=none, 1=every other, 2=every 3rd, etc.
  gchar *embedding_classes;  // Semicolon-separated list of classes to generate embeddings for (NULL = all classes)
  gboolean query_only;  // Init model but skip per-frame inference; REST queries still work
  
  /* Runtime state */
  guint frame_count;
  gchar **allowed_classes;  // Parsed array of allowed class names
  guint num_allowed_classes;  // Number of allowed classes
  gboolean is_initialized;
  
  /* Triton client handle (C++ object wrapped as void*) */
  void *triton_client;

  /* Serializes triton_client across streaming, REST query and the OTA commit */
  GMutex infer_mutex;

  /* Runtime model update (OTA). Lock order: replacement_lock -> infer_mutex */
  GMutex       config_lock;        // guards the model config strings above
  GMutex       replacement_lock;   // guards next_replacement / the flags below
  GThread     *model_load_thread;  // background engine deserialize
  GAsyncQueue *pending_models;     // NvDsVisionEncoderLoadRequest*
  gboolean     loader_running;     // FALSE once stop() begins
  gboolean     update_in_progress;  // an update is in flight: reject new ones
  NvDsVisionEncoderReplacement *next_replacement;  // built, validated, not live
  gchar       *current_model_path; // what is serving now
  
  /* CUDA stream for async operations */
  void *cuda_stream;  // cudaStream_t
  
  /* Model info */
  guint input_width;
  guint input_height;
  guint embedding_dim;
  gboolean use_siglip_norm;  // resolved at start(); immutable (no cross-family update)
  
  /* Pre-allocated buffers for preprocessing (like nvinfer SGIE) */
  guint max_batch_size;  // Maximum number of objects per transform call
  NvBufSurfaceParams *prealloc_surf_list;  // Pre-allocated surface list for src_batch
  NvBufSurfTransformRect *prealloc_src_rect;  // Pre-allocated source rectangles
  NvBufSurfTransformRect *prealloc_dst_rect;  // Pre-allocated destination rectangles
  NvBufSurface *prealloc_output_surface;  // Pre-allocated output surface (reused across calls)
  float *d_preprocessed_pool;  // Pre-allocated device tensor (max_batch_size) for GPU path
  size_t d_preprocessed_pool_bytes;  // Allocated size in bytes
  float *embeddings_buffer;    // Pre-allocated host buffer for inference output (max_batch_size * embedding_dim)
  
  /* Smart inference: cache embeddings by tracker object_id to avoid redundant inference */
  gboolean smart_infer;
  guint cache_refresh_interval;
  guint cache_max_size;
  GHashTable *embedding_cache;      /* guint64 object_id -> CachedEmbedding* */
  guint64 global_frame_counter;     /* Monotonic frame counter for cache eviction */

  /*
   * OFA-based temporal embedding prediction.
   * Uses hardware optical flow motion vectors from upstream nvof element to predict
   * whether a cached embedding is still valid, and to transform stale embeddings
   * using motion-derived features instead of running full inference.
   */
  gboolean ofa_predict_enabled;
  gfloat ofa_motion_threshold;      /* Below this: embedding is stable, skip inference */
  gfloat ofa_high_motion_threshold; /* Above this: force re-inference (drastic change) */
  guint64 ofa_predictions_total;    /* Stats: total motion-predicted embeddings */
  guint64 ofa_reinfers_total;       /* Stats: total motion-triggered re-inferences */

  /*
   * DLA two-tier inference.
   * A lightweight "appearance change detector" runs on DLA (or GPU fallback) to decide
   * whether the full vision encoder needs to re-run for a tracked object.
   * Tier 1 (DLA/GPU): small binary classifier: "appearance changed?" on 32x32 crop
   * Tier 2 (GPU): full vision encoder on cache-miss + validator-flagged objects
   */
  gboolean dla_validator_enabled;
  gchar *dla_validator_engine;      /* Path to TensorRT engine for the validator */
  gint dla_core;                    /* DLA core index (-1 = GPU fallback) */
  gfloat dla_change_threshold;      /* Probability threshold for "appearance changed" */
  DlaValidatorCtx *dla_ctx;
  guint64 dla_validations_total;
  guint64 dla_changes_detected;

#ifdef ENABLE_INSTRUMENTATION
  /* Performance counters */
  GstClockTime total_process_time;
  guint total_objects_processed;

  /* Periodic performance reporting (every PERF_WINDOW_SEC seconds) */
  struct timeval perf_window_start;
  guint perf_window_objects;
  guint perf_window_batches;
  gboolean perf_window_started;

  /* Per-phase cumulative microseconds within current window */
  guint64 perf_phase_preproc_us;   /* NvBufSurfTransform crop+resize */
  guint64 perf_phase_tensor_us;    /* RGB surface -> CHW float tensor */
  guint64 perf_phase_infer_us;     /* TRT inference + D2H memcpy */
  guint64 perf_phase_post_us;      /* Attach embedding metadata */
#endif
};

/**
 * GstNvDsVisionEncoderClass structure
 */
struct _GstNvDsVisionEncoderClass {
  GstBaseTransformClass parent_class;

  /* "model-updated": same signature as gst-nvinfer's, (gint status, gchar *path),
   * status 0 = OK, so nvmodelmux can connect one callback to either element. */
  void (*model_updated) (GstNvDsVisionEncoder *encoder, gint status,
      gchar *path);
};

GType gst_nvdsvisionencoder_get_type (void);

G_END_DECLS

#endif /* __GST_NVDSVISIONENCODER_H__ */
