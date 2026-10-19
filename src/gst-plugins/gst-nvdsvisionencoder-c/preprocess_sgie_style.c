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

#include <string.h>
#include <stdio.h>
#include <glib.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gstnvdsvisionencoder.h"
#include "preprocess.h"

/**
 * Preprocess crops using NvBufSurfTransform (like nvinfer SGIE)
 * Uses pre-allocated buffers and output surface for better performance
 * Returns 0 on success, -1 on error
 */
int
preprocess_with_nvbufsurftransform(
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
    NvBufSurfTransformRect *prealloc_dst_rect)
{
    if (num_objects == 0) {
        return -1;
    }
    
    if (num_objects > max_batch_size) {
        g_print("[SGIE-PREPROCESS] ERROR: num_objects(%u) exceeds max_batch_size(%u)\n",
                num_objects, max_batch_size);
        return -1;
    }
    
    /* Set CUDA device for preprocessing operations */
    cudaError_t cuda_err = cudaSetDevice(gpu_id);
    if (cuda_err != cudaSuccess) {
        g_print("[SGIE-PREPROCESS] Failed to set CUDA device %u: %s\n", 
                gpu_id, cudaGetErrorString(cuda_err));
        return -1;
    }
    
    /* Reuse pre-allocated output surface (like nvinfer SGIE buffer pool) */
    /* No need to create/destroy - surface persists across calls */
    
    /* Get source surface dimensions */
    guint src_width = input_surface->surfaceList[batch_id].width;
    guint src_height = input_surface->surfaceList[batch_id].height;
    
    /* g_print("[SGIE-PREPROCESS] Input surface[%u]: %ux%u, colorFormat=%u\n",
            batch_id, src_width, src_height, 
            input_surface->surfaceList[batch_id].colorFormat); */
    
    /* Setup source and destination batches using pre-allocated buffers (like nvinfer) */
    NvBufSurface src_batch, dst_batch;
    memset(&src_batch, 0, sizeof(NvBufSurface));
    memset(&dst_batch, 0, sizeof(NvBufSurface));
    
    src_batch.gpuId = input_surface->gpuId;
    src_batch.batchSize = num_objects;
    src_batch.numFilled = num_objects;
    src_batch.memType = input_surface->memType;
    src_batch.isContiguous = FALSE; /* Required for aarch64 VIC */
    src_batch.surfaceList = prealloc_surf_list;  /* Use pre-allocated */
    
    dst_batch.gpuId = prealloc_output_surface->gpuId;
    dst_batch.batchSize = num_objects;
    dst_batch.numFilled = num_objects;
    dst_batch.memType = prealloc_output_surface->memType;
    dst_batch.isContiguous = prealloc_output_surface->isContiguous; /* Required for aarch64 VIC */
    dst_batch.surfaceList = prealloc_output_surface->surfaceList;  /* Use pre-allocated */
    
    /* Use pre-allocated transform rectangles (like nvinfer) */
    NvBufSurfTransformRect *src_rect = prealloc_src_rect;
    NvBufSurfTransformRect *dst_rect = prealloc_dst_rect;
    
    /* Fill transform parameters for each object */
    /* g_print("[SGIE-PREPROCESS] Processing %u objects from batch_id=%u, src surface: %ux%u\n",
            num_objects, batch_id, src_width, src_height); */
    
    for (guint i = 0; i < num_objects; i++) {
        /* All crops from same source frame */
        src_batch.surfaceList[i] = input_surface->surfaceList[batch_id];
        
        /* Get object bbox */
        NvOSD_RectParams *rect = &obj_meta_array[i]->rect_params;
        
        guint left = (guint)rect->left;
        guint top = (guint)rect->top;
        guint width = (guint)rect->width;
        guint height = (guint)rect->height;
        
        /* g_print("[SGIE-PREPROCESS] Obj %u: Original bbox: left=%u, top=%u, width=%u, height=%u\n",
                i, left, top, width, height); */
        
        /* Clamp to surface boundaries (safety) */
        if (left >= src_width) left = src_width - 1;
        if (top >= src_height) top = src_height - 1;
        if (left + width > src_width) width = src_width - left;
        if (top + height > src_height) height = src_height - top;
        
        /* Ensure minimum size */
        if (width < 1) width = 1;
        if (height < 1) height = 1;
        
        /* g_print("[SGIE-PREPROCESS] Obj %u: Clamped bbox: left=%u, top=%u, width=%u, height=%u\n",
                i, left, top, width, height); */
        
        /* Source ROI (crop from frame) */
        src_rect[i].top = top;
        src_rect[i].left = left;
        src_rect[i].width = width;
        src_rect[i].height = height;
        
        /* Destination ROI (full output surface) */
        dst_rect[i].top = 0;
        dst_rect[i].left = 0;
        dst_rect[i].width = target_width;
        dst_rect[i].height = target_height;
        
        /* g_print("[SGIE-PREPROCESS] Obj %u: src_rect=[%u,%u,%ux%u], dst_rect=[%u,%u,%ux%u]\n",
                i, src_rect[i].left, src_rect[i].top, src_rect[i].width, src_rect[i].height,
                dst_rect[i].left, dst_rect[i].top, dst_rect[i].width, dst_rect[i].height); */
        
        /* Print surface info */
        /* NvBufSurfaceParams *surf = &src_batch.surfaceList[i];
        g_print("[SGIE-PREPROCESS] Obj %u: Surface: %ux%u, pitch=%u, dataPtr=%p, colorFormat=%u\n",
                i, surf->width, surf->height, surf->pitch, surf->dataPtr, surf->colorFormat); */
    }
    
    /* Setup transform parameters (same as nvinfer SGIE) */
    NvBufSurfTransformParams transform_params;
    memset(&transform_params, 0, sizeof(NvBufSurfTransformParams));
    transform_params.src_rect = src_rect;
    transform_params.dst_rect = dst_rect;
    transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER | 
                                      NVBUFSURF_TRANSFORM_CROP_SRC | 
                                      NVBUFSURF_TRANSFORM_CROP_DST |
                                      NVBUFSURF_TRANSFORM_ALLOW_ODD_CROP;
    transform_params.transform_filter = NvBufSurfTransformInter_Default;
    transform_params.transform_flip = NvBufSurfTransform_None;
    
    /* Perform hardware-accelerated transformation (like nvinfer SGIE) */
    /* NvBufSurfTransform handles surface mapping internally */
    /* g_print("[SGIE-PREPROCESS] Calling NvBufSurfTransform with %u crops\n", num_objects); */
    int ret = NvBufSurfTransform(&src_batch, &dst_batch, &transform_params);
    /* g_print("[SGIE-PREPROCESS] NvBufSurfTransform returned: %d\n", ret); */
    
    /* No cleanup needed - using pre-allocated buffers that persist across calls */
    
    if (ret != 0) {
        g_print("[SGIE-PREPROCESS] NvBufSurfTransform failed: %d\n", ret);
        return -1;
    }
    
    /* Sync output surface to ensure transform completed before CPU access */
    NvBufSurfaceSyncForCpu(prealloc_output_surface, -1, -1);
    /* g_print("[SGIE-PREPROCESS] Output surface synced for CPU access\n"); */
    
    return 0;
}

/**
 * Single-call batch preprocess (SGIE-style): all objects from all frames
 * in one NvBufSurfTransform. Each slot i uses source frame surfaceList[batch_ids[i]].
 * Returns 0 on success, -1 on error.
 */
int
preprocess_batch_with_nvbufsurftransform(
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
    gboolean output_for_gpu)
{
    if (num_objects == 0) {
        return -1;
    }

    if (num_objects > max_batch_size) {
        g_print("[SGIE-PREPROCESS] ERROR: num_objects(%u) exceeds max_batch_size(%u)\n",
                num_objects, max_batch_size);
        return -1;
    }

    cudaError_t cuda_err = cudaSetDevice(gpu_id);
    if (cuda_err != cudaSuccess) {
        g_print("[SGIE-PREPROCESS] Failed to set CUDA device %u: %s\n",
                gpu_id, cudaGetErrorString(cuda_err));
        return -1;
    }

    /* Source batch: each slot points to the correct input frame (like nvinfer SGIE) */
    for (guint i = 0; i < num_objects; i++) {
        guint bid = batch_ids[i];
        prealloc_surf_list[i] = input_surface->surfaceList[bid];
    }

    NvBufSurface src_batch, dst_batch;
    memset(&src_batch, 0, sizeof(NvBufSurface));
    memset(&dst_batch, 0, sizeof(NvBufSurface));

    src_batch.gpuId = input_surface->gpuId;
    src_batch.batchSize = num_objects;
    src_batch.numFilled = num_objects;
    src_batch.memType = input_surface->memType;
    src_batch.isContiguous = FALSE;  /* Required for aarch64 VIC */
    src_batch.surfaceList = prealloc_surf_list;

    dst_batch.gpuId = prealloc_output_surface->gpuId;
    dst_batch.batchSize = num_objects;
    dst_batch.numFilled = num_objects;
    dst_batch.memType = prealloc_output_surface->memType;
    dst_batch.isContiguous = prealloc_output_surface->isContiguous;  /* Required for aarch64 VIC */
    dst_batch.surfaceList = prealloc_output_surface->surfaceList;

    NvBufSurfTransformRect *src_rect = prealloc_src_rect;
    NvBufSurfTransformRect *dst_rect = prealloc_dst_rect;

    for (guint i = 0; i < num_objects; i++) {
        guint bid = batch_ids[i];
        guint src_width = input_surface->surfaceList[bid].width;
        guint src_height = input_surface->surfaceList[bid].height;

        NvOSD_RectParams *rect = &obj_meta_array[i]->rect_params;
        guint left = (guint)rect->left;
        guint top = (guint)rect->top;
        guint width = (guint)rect->width;
        guint height = (guint)rect->height;

        if (left >= src_width) left = src_width - 1;
        if (top >= src_height) top = src_height - 1;
        if (left + width > src_width) width = src_width - left;
        if (top + height > src_height) height = src_height - top;
        if (width < 1) width = 1;
        if (height < 1) height = 1;

        src_rect[i].top = top;
        src_rect[i].left = left;
        src_rect[i].width = width;
        src_rect[i].height = height;

        dst_rect[i].top = 0;
        dst_rect[i].left = 0;
        dst_rect[i].width = target_width;
        dst_rect[i].height = target_height;
    }

    NvBufSurfTransformParams transform_params;
    memset(&transform_params, 0, sizeof(NvBufSurfTransformParams));
    transform_params.src_rect = src_rect;
    transform_params.dst_rect = dst_rect;
    transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER |
                                      NVBUFSURF_TRANSFORM_CROP_SRC |
                                      NVBUFSURF_TRANSFORM_CROP_DST |
                                      NVBUFSURF_TRANSFORM_ALLOW_ODD_CROP;
    transform_params.transform_filter = NvBufSurfTransformInter_Default;
    transform_params.transform_flip = NvBufSurfTransform_None;

    int ret = NvBufSurfTransform(&src_batch, &dst_batch, &transform_params);
    if (ret != 0) {
        g_print("[SGIE-PREPROCESS] NvBufSurfTransform (batch) failed: %d\n", ret);
        return -1;
    }

    /* Sync for next consumer: device path uses GPU only (no D2H); host path needs CPU sync */
    if (output_for_gpu) {
        NvBufSurfaceSyncForDevice(prealloc_output_surface, -1, -1);
    } else {
        NvBufSurfaceSyncForCpu(prealloc_output_surface, -1, -1);
    }
    return 0;
}

/**
 * Convert RGB surface to normalized float tensor (CHW format)
 * Simple, safe CUDA operation - just normalization
 */
int
rgb_surface_to_tensor(
    NvBufSurface *rgb_surface,
    float *output_tensor,
    guint num_images,
    guint width,
    guint height,
    gboolean use_siglip_normalization)
{
    /* Set CUDA device based on surface's GPU */
    cudaError_t cuda_err = cudaSetDevice(rgb_surface->gpuId);
    if (cuda_err != cudaSuccess) {
        g_print("[SGIE-PREPROCESS] Failed to set CUDA device %u: %s\n",
                rgb_surface->gpuId, cudaGetErrorString(cuda_err));
        return -1;
    }
    
    /* Map surface for CPU access to read data */
    if (NvBufSurfaceMap(rgb_surface, -1, -1, NVBUF_MAP_READ) != 0) {
        g_print("[SGIE-PREPROCESS] Failed to map RGB surface for reading\n");
        return -1;
    }
    
    /* Sync to ensure data is available for CPU access */
    NvBufSurfaceSyncForCpu(rgb_surface, -1, -1);
    
    for (guint img_idx = 0; img_idx < num_images; img_idx++) {
        NvBufSurfaceParams *surf_params = &rgb_surface->surfaceList[img_idx];
        
        /* Get GPU pointer directly - no CPU mapping needed */
        unsigned char *rgb_gpu_data = (unsigned char *)surf_params->dataPtr;
        guint pitch = surf_params->pitch;
        
        /* RADIO-CLIP uses simple [0,1] normalization (no ImageNet mean/std) */
        
        /* Allocate temporary host buffer */
        guint hw = height * width;
        unsigned char *rgb_host = (unsigned char *)g_malloc(height * pitch);
        
        /* Copy from device to host */
        if (cudaMemcpy(rgb_host, rgb_gpu_data, height * pitch, cudaMemcpyDeviceToHost) != cudaSuccess) {
            g_print("[SGIE-PREPROCESS] cudaMemcpy failed\n");
            g_free(rgb_host);
            NvBufSurfaceUnMap(rgb_surface, -1, -1);
            return -1;
        }
        
        /* Convert to CHW: [0,1] (RADIO-CLIP) or [-1,1] SigLIP/SigLIP2 */
        for (guint c = 0; c < 3; c++) {
            for (guint h = 0; h < height; h++) {
                for (guint w = 0; w < width; w++) {
                    guint rgb_offset = h * pitch + w * 3 + c;
                    float val = rgb_host[rgb_offset] / 255.0f;
                    if (use_siglip_normalization)
                        val = (val - 0.5f) / 0.5f;  /* SigLIP [-1,1] */
                    guint out_offset = img_idx * (3 * hw) + c * hw + h * width + w;
                    output_tensor[out_offset] = val;
                }
            }
        }
        
        g_free(rgb_host);
    }
    
    /* Unmap RGB surface */
    NvBufSurfaceUnMap(rgb_surface, -1, -1);
    
    return 0;
}

/**
 * Convert RGBA surface to normalized float tensor (CHW format)
 * Similar to rgb_surface_to_tensor but handles RGBA (4 bytes per pixel)
 */
int
rgba_surface_to_tensor(
    NvBufSurface *rgba_surface,
    float *output_tensor,
    guint num_images,
    guint width,
    guint height,
    gboolean use_siglip_normalization)
{
    /* Set CUDA device based on surface's GPU */
    cudaError_t cuda_err = cudaSetDevice(rgba_surface->gpuId);
    if (cuda_err != cudaSuccess) {
        g_print("[SGIE-PREPROCESS] Failed to set CUDA device %u: %s\n",
                rgba_surface->gpuId, cudaGetErrorString(cuda_err));
        return -1;
    }
    
    /* Map surface for CPU access to read data */
    if (NvBufSurfaceMap(rgba_surface, -1, -1, NVBUF_MAP_READ) != 0) {
        g_print("[SGIE-PREPROCESS] Failed to map RGBA surface for reading\n");
        return -1;
    }
    
    /* Sync to ensure data is available for CPU access */
    NvBufSurfaceSyncForCpu(rgba_surface, -1, -1);
    
    for (guint img_idx = 0; img_idx < num_images; img_idx++) {
        NvBufSurfaceParams *surf_params = &rgba_surface->surfaceList[img_idx];
        
        /* Get GPU pointer directly - no CPU mapping needed */
        unsigned char *rgba_gpu_data = (unsigned char *)surf_params->dataPtr;
        guint pitch = surf_params->pitch;
        
        /* Allocate temporary host buffer */
        guint hw = height * width;
        unsigned char *rgba_host = (unsigned char *)g_malloc(height * pitch);
        
        /* Copy from device to host */
        if (cudaMemcpy(rgba_host, rgba_gpu_data, height * pitch, cudaMemcpyDeviceToHost) != cudaSuccess) {
            g_print("[SGIE-PREPROCESS] cudaMemcpy failed\n");
            g_free(rgba_host);
            NvBufSurfaceUnMap(rgba_surface, -1, -1);
            return -1;
        }
        
        /* Convert to CHW: [0,1] (RADIO-CLIP) or [-1,1] SigLIP/SigLIP2 - ignore alpha channel */
        for (guint c = 0; c < 3; c++) {
            for (guint h = 0; h < height; h++) {
                for (guint w = 0; w < width; w++) {
                    guint rgba_offset = h * pitch + w * 4 + c;  /* 4 bytes per pixel for RGBA */
                    float val = rgba_host[rgba_offset] / 255.0f;
                    if (use_siglip_normalization)
                        val = (val - 0.5f) / 0.5f;  /* SigLIP [-1,1] */
                    guint out_offset = img_idx * (3 * hw) + c * hw + h * width + w;
                    output_tensor[out_offset] = val;
                }
            }
        }
        
        g_free(rgba_host);
    }
    
    /* Unmap RGBA surface */
    NvBufSurfaceUnMap(rgba_surface, -1, -1);
    
    return 0;
}
