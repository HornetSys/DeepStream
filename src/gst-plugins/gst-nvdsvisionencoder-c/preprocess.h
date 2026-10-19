/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __PREPROCESS_H__
#define __PREPROCESS_H__

#include <cuda_runtime.h>
#include <glib.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvdsmeta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * NEW SGIE-style preprocessing using NvBufSurfTransform
 * Uses pre-allocated buffers and output surface (like nvinfer SGIE does)
 */
int preprocess_with_nvbufsurftransform(
    NvBufSurface *input_surface,
    guint batch_id,
    NvDsObjectMeta **obj_meta_array,
    guint num_objects,
    NvBufSurface *prealloc_output_surface,
    guint target_width,
    guint target_height,
    guint gpu_id,
    guint max_batch_size,
    NvBufSurfaceParams *prealloc_surf_list,
    NvBufSurfTransformRect *prealloc_src_rect,
    NvBufSurfTransformRect *prealloc_dst_rect);

/**
 * Single-call batch preprocess (SGIE-style): all objects from all frames
 * in one NvBufSurfTransform. batch_ids[i] = frame batch_id for object i.
 * output_for_gpu: if TRUE, call SyncForDevice (next step is GPU); if FALSE, SyncForCpu (avoids unnecessary D2H sync on device path).
 */
int preprocess_batch_with_nvbufsurftransform(
    NvBufSurface *input_surface,
    const guint *batch_ids,
    NvDsObjectMeta **obj_meta_array,
    guint num_objects,
    NvBufSurface *prealloc_output_surface,
    guint target_width,
    guint target_height,
    guint gpu_id,
    guint max_batch_size,
    NvBufSurfaceParams *prealloc_surf_list,
    NvBufSurfTransformRect *prealloc_src_rect,
    NvBufSurfTransformRect *prealloc_dst_rect,
    gboolean output_for_gpu);

int rgb_surface_to_tensor(
    NvBufSurface *rgb_surface,
    float *output_tensor,
    guint num_images,
    guint width,
    guint height,
    gboolean use_siglip_normalization);

/** GPU path: write tensor on device (no D2H). Use with nvds_triton_client_infer_device for FPS. */
int rgb_surface_to_tensor_device(
    NvBufSurface *rgb_surface,
    float *d_output_tensor,
    guint num_images,
    guint width,
    guint height,
    gboolean use_siglip_normalization);

int rgba_surface_to_tensor(
    NvBufSurface *rgba_surface,
    float *output_tensor,
    guint num_images,
    guint width,
    guint height,
    gboolean use_siglip_normalization);

/** GPU path for RGBA: write tensor on device (no D2H). Use with nvds_triton_client_infer_device for FPS. */
int rgba_surface_to_tensor_device(
    NvBufSurface *rgba_surface,
    float *d_output_tensor,
    guint num_images,
    guint width,
    guint height,
    gboolean use_siglip_normalization);

/**
 * LEGACY: Preprocess multiple crops from an RGBA surface
 */
int preprocess_crops_rgba(
    const void* rgba_surface,
    int surface_width,
    int surface_height,
    int surface_pitch,
    const float* crop_boxes,
    int num_crops,
    float* output,
    int target_width,
    int target_height,
    cudaStream_t stream
);

/**
 * Preprocess multiple crops from an NV12 surface
 * 
 * @param nv12_surface Device pointer to NV12 surface
 * @param surface_width Width of the surface
 * @param surface_height Height of the surface
 * @param surface_pitch Pitch (stride) of the surface
 * @param crop_boxes Host array of crop boxes [num_crops, 4] (left, top, width, height)
 * @param num_crops Number of crops
 * @param output Device pointer for output [num_crops, 3, target_h, target_w]
 * @param target_width Target width for each crop
 * @param target_height Target height for each crop
 * @param stream CUDA stream
 * @return 0 on success, -1 on error
 */
int preprocess_crops_nv12(
    const void* nv12_surface,
    int surface_width,
    int surface_height,
    int surface_pitch,
    const float* crop_boxes,
    int num_crops,
    float* output,
    int target_width,
    int target_height,
    cudaStream_t stream
);

/**
 * Map and get device pointer from NvBufSurface
 * Note: Caller must call unmap_nvbufsurface after use
 */
void* get_nvbufsurface_device_ptr(NvBufSurface* surf, unsigned int buf_idx);

/**
 * Unmap NvBufSurface after use
 */
void unmap_nvbufsurface(NvBufSurface* surf, unsigned int buf_idx);

#ifdef __cplusplus
}
#endif

#endif // __PREPROCESS_H__
