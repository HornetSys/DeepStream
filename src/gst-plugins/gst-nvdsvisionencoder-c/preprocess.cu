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

#include <cuda_runtime.h>
#include <stdio.h>
#include <stddef.h>
#include <cstdlib>
#include "nvbufsurface.h"
#include "preprocess.h"

/**
 * Kernel: RGB image (pitch layout, u8) -> CHW float [0,1] on device.
 * Used for full GPU path to avoid D2H copy and CPU loop (FPS optimization).
 */
__global__ void rgb_pitch_to_chw_float_kernel(
    const unsigned char* __restrict__ rgb,
    int pitch,
    int width,
    int height,
    float* __restrict__ d_output,
    size_t output_offset_floats
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int hw = width * height;
    size_t base = output_offset_floats;
    int src = y * pitch + x * 3;
    float r = rgb[src + 0] / 255.0f;
    float g = rgb[src + 1] / 255.0f;
    float b = rgb[src + 2] / 255.0f;
    d_output[base + 0 * hw + y * width + x] = r;
    d_output[base + 1 * hw + y * width + x] = g;
    d_output[base + 2 * hw + y * width + x] = b;
}

struct SurfaceParams { unsigned char* data; int pitch; };

/** Batched version: one launch for all images. use_siglip: 0 = [0,1], 1 = [-1,1] SigLIP. */
__global__ void rgb_surface_batch_to_chw_float_kernel(
    const SurfaceParams* __restrict__ params,
    int num_images,
    int width,
    int height,
    int use_siglip,
    float* __restrict__ d_output
) {
    int img = blockIdx.z;
    if (img >= num_images) return;
    const unsigned char* rgb = params[img].data;
    int pitch = params[img].pitch;
    size_t floats_per_image = (size_t)3 * width * height;
    size_t base = (size_t)img * floats_per_image;
    int hw = width * height;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int src = y * pitch + x * 3;
    float r = rgb[src + 0] / 255.0f;
    float g = rgb[src + 1] / 255.0f;
    float b = rgb[src + 2] / 255.0f;
    if (use_siglip) {
        r = (r - 0.5f) / 0.5f;
        g = (g - 0.5f) / 0.5f;
        b = (b - 0.5f) / 0.5f;
    }
    d_output[base + 0 * hw + y * width + x] = r;
    d_output[base + 1 * hw + y * width + x] = g;
    d_output[base + 2 * hw + y * width + x] = b;
}

/** Batched version for RGBA: one launch for all images. use_siglip: 0 = [0,1], 1 = [-1,1] SigLIP. */
__global__ void rgba_surface_batch_to_chw_float_kernel(
    const SurfaceParams* __restrict__ params,
    int num_images,
    int width,
    int height,
    int use_siglip,
    float* __restrict__ d_output
) {
    int img = blockIdx.z;
    if (img >= num_images) return;
    const unsigned char* rgba = params[img].data;
    int pitch = params[img].pitch;
    size_t floats_per_image = (size_t)3 * width * height;
    size_t base = (size_t)img * floats_per_image;
    int hw = width * height;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int src = y * pitch + x * 4;  /* 4 bytes per pixel for RGBA */
    float r = rgba[src + 0] / 255.0f;
    float g = rgba[src + 1] / 255.0f;
    float b = rgba[src + 2] / 255.0f;
    /* Ignore alpha channel */
    if (use_siglip) {
        r = (r - 0.5f) / 0.5f;
        g = (g - 0.5f) / 0.5f;
        b = (b - 0.5f) / 0.5f;
    }
    d_output[base + 0 * hw + y * width + x] = r;
    d_output[base + 1 * hw + y * width + x] = g;
    d_output[base + 2 * hw + y * width + x] = b;
}

// Kernel to extract and preprocess crops from RGBA surface
__global__ void preprocess_rgba_crops_kernel(
    const unsigned char* rgba_surface,
    int surface_width,
    int surface_height,
    int surface_pitch,
    const float* crop_boxes,  // [num_crops, 4] - left, top, width, height
    int num_crops,
    float* output,           // [num_crops, 3, 224, 224]
    int target_width,
    int target_height
) {
    int crop_idx = blockIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (crop_idx >= num_crops || out_y >= target_height || out_x >= target_width) {
        return;
    }
    
    // Get crop coordinates
    float crop_left = crop_boxes[crop_idx * 4 + 0];
    float crop_top = crop_boxes[crop_idx * 4 + 1];
    float crop_width = crop_boxes[crop_idx * 4 + 2];
    float crop_height = crop_boxes[crop_idx * 4 + 3];
    
    // Clamp crop to surface boundaries
    crop_left = fmaxf(0.0f, fminf(crop_left, (float)(surface_width - 1)));
    crop_top = fmaxf(0.0f, fminf(crop_top, (float)(surface_height - 1)));
    crop_width = fminf(crop_width, (float)surface_width - crop_left);
    crop_height = fminf(crop_height, (float)surface_height - crop_top);
    
    // Safety check: skip invalid crops
    if (crop_width < 1.0f || crop_height < 1.0f) {
        // Write black pixel for invalid crops
        int out_offset = crop_idx * (3 * target_height * target_width);
        int hw = target_height * target_width;
        output[out_offset + 0 * hw + out_y * target_width + out_x] = 0.0f;
        output[out_offset + 1 * hw + out_y * target_width + out_x] = 0.0f;
        output[out_offset + 2 * hw + out_y * target_width + out_x] = 0.0f;
        return;
    }
    
    // Map output pixel to input crop pixel
    float src_x = crop_left + (out_x + 0.5f) * crop_width / target_width;
    float src_y = crop_top + (out_y + 0.5f) * crop_height / target_height;
    
    // Bilinear interpolation coordinates with bounds checking
    int x0 = max(0, min((int)floorf(src_x), surface_width - 1));
    int y0 = max(0, min((int)floorf(src_y), surface_height - 1));
    int x1 = min(x0 + 1, surface_width - 1);
    int y1 = min(y0 + 1, surface_height - 1);
    
    float wx = src_x - x0;
    float wy = src_y - y0;
    
    // Additional safety: verify coordinates are within bounds
    if (x0 < 0 || x1 >= surface_width || y0 < 0 || y1 >= surface_height) {
        // Write black pixel for safety
        int out_offset = crop_idx * (3 * target_height * target_width);
        int hw = target_height * target_width;
        output[out_offset + 0 * hw + out_y * target_width + out_x] = 0.0f;
        output[out_offset + 1 * hw + out_y * target_width + out_x] = 0.0f;
        output[out_offset + 2 * hw + out_y * target_width + out_x] = 0.0f;
        return;
    }
    
    // Sample RGBA pixels (4 bytes per pixel)
    int offset00 = y0 * surface_pitch + x0 * 4;
    int offset01 = y0 * surface_pitch + x1 * 4;
    int offset10 = y1 * surface_pitch + x0 * 4;
    int offset11 = y1 * surface_pitch + x1 * 4;
    
    // Bilinear interpolation for R, G, B (ignore A)
    float R = (1 - wx) * (1 - wy) * rgba_surface[offset00 + 0] +
              wx * (1 - wy) * rgba_surface[offset01 + 0] +
              (1 - wx) * wy * rgba_surface[offset10 + 0] +
              wx * wy * rgba_surface[offset11 + 0];
              
    float G = (1 - wx) * (1 - wy) * rgba_surface[offset00 + 1] +
              wx * (1 - wy) * rgba_surface[offset01 + 1] +
              (1 - wx) * wy * rgba_surface[offset10 + 1] +
              wx * wy * rgba_surface[offset11 + 1];
              
    float B = (1 - wx) * (1 - wy) * rgba_surface[offset00 + 2] +
              wx * (1 - wy) * rgba_surface[offset01 + 2] +
              (1 - wx) * wy * rgba_surface[offset10 + 2] +
              wx * wy * rgba_surface[offset11 + 2];
    
    // Normalize to [0, 1] only (RADIO-CLIP expects simple [0,1] normalization)
    R /= 255.0f;
    G /= 255.0f;
    B /= 255.0f;
    
    // No ImageNet mean/std normalization for RADIO-CLIP
    
    // Write to output in CHW format: [C, H, W]
    int out_offset = crop_idx * (3 * target_height * target_width);
    int hw = target_height * target_width;
    
    output[out_offset + 0 * hw + out_y * target_width + out_x] = R;
    output[out_offset + 1 * hw + out_y * target_width + out_x] = G;
    output[out_offset + 2 * hw + out_y * target_width + out_x] = B;
}

// Kernel to extract and preprocess crops from NV12 surface (legacy - not used)
__global__ void preprocess_nv12_crops_kernel(
    const unsigned char* nv12_surface,
    int surface_width,
    int surface_height,
    int surface_pitch,
    const float* crop_boxes,  // [num_crops, 4] - left, top, width, height
    int num_crops,
    float* output,           // [num_crops, 3, 224, 224]
    int target_width,
    int target_height
) {
    int crop_idx = blockIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (crop_idx >= num_crops || out_y >= target_height || out_x >= target_width) {
        return;
    }
    
    // Get crop coordinates
    float crop_left = crop_boxes[crop_idx * 4 + 0];
    float crop_top = crop_boxes[crop_idx * 4 + 1];
    float crop_width = crop_boxes[crop_idx * 4 + 2];
    float crop_height = crop_boxes[crop_idx * 4 + 3];
    
    // Clamp crop to surface boundaries
    crop_left = fmaxf(0.0f, fminf(crop_left, (float)surface_width - 1));
    crop_top = fmaxf(0.0f, fminf(crop_top, (float)surface_height - 1));
    crop_width = fminf(crop_width, (float)surface_width - crop_left);
    crop_height = fminf(crop_height, (float)surface_height - crop_top);
    
    // Safety check: skip invalid crops
    if (crop_width < 1.0f || crop_height < 1.0f) {
        // Write black pixel for invalid crops
        int out_offset = crop_idx * (3 * target_height * target_width);
        int hw = target_height * target_width;
        output[out_offset + 0 * hw + out_y * target_width + out_x] = 0.0f;
        output[out_offset + 1 * hw + out_y * target_width + out_x] = 0.0f;
        output[out_offset + 2 * hw + out_y * target_width + out_x] = 0.0f;
        return;
    }
    
    // Map output pixel to input crop pixel
    float src_x = crop_left + (out_x + 0.5f) * crop_width / target_width;
    float src_y = crop_top + (out_y + 0.5f) * crop_height / target_height;
    
    // Bilinear interpolation coordinates with bounds checking
    int x0 = max(0, min((int)floorf(src_x), surface_width - 1));
    int y0 = max(0, min((int)floorf(src_y), surface_height - 1));
    int x1 = min(x0 + 1, surface_width - 1);
    int y1 = min(y0 + 1, surface_height - 1);
    
    // Additional safety: verify coordinates are within bounds
    if (x0 < 0 || x1 >= surface_width || y0 < 0 || y1 >= surface_height) {
        // Write black pixel for safety
        int out_offset = crop_idx * (3 * target_height * target_width);
        int hw = target_height * target_width;
        output[out_offset + 0 * hw + out_y * target_width + out_x] = 0.0f;
        output[out_offset + 1 * hw + out_y * target_width + out_x] = 0.0f;
        output[out_offset + 2 * hw + out_y * target_width + out_x] = 0.0f;
        return;
    }
    
    float wx = src_x - x0;
    float wy = src_y - y0;
    
    // Sample Y plane (luma)
    int y_offset0 = y0 * surface_pitch + x0;
    int y_offset1 = y0 * surface_pitch + x1;
    int y_offset2 = y1 * surface_pitch + x0;
    int y_offset3 = y1 * surface_pitch + x1;
    
    float Y = (1 - wx) * (1 - wy) * nv12_surface[y_offset0] +
              wx * (1 - wy) * nv12_surface[y_offset1] +
              (1 - wx) * wy * nv12_surface[y_offset2] +
              wx * wy * nv12_surface[y_offset3];
    
    // Sample UV plane (chroma) - interleaved U and V, subsampled 2x2
    int uv_plane_offset = surface_pitch * surface_height;
    int uv_y = y0 / 2;
    int uv_x = (x0 / 2) * 2;  // Ensure even for U/V pairs
    
    int uv_offset = uv_plane_offset + uv_y * surface_pitch + uv_x;
    float U = nv12_surface[uv_offset];
    float V = nv12_surface[uv_offset + 1];
    
    // Convert YUV to RGB (BT.601 standard)
    float R = Y + 1.402f * (V - 128.0f);
    float G = Y - 0.344f * (U - 128.0f) - 0.714f * (V - 128.0f);
    float B = Y + 1.772f * (U - 128.0f);
    
    // Clamp to [0, 255]
    R = fmaxf(0.0f, fminf(255.0f, R));
    G = fmaxf(0.0f, fminf(255.0f, G));
    B = fmaxf(0.0f, fminf(255.0f, B));
    
    // Normalize to [0, 1] only (RADIO-CLIP expects simple [0,1] normalization)
    R /= 255.0f;
    G /= 255.0f;
    B /= 255.0f;
    
    // No ImageNet mean/std normalization for RADIO-CLIP
    
    // Write to output in CHW format: [C, H, W]
    int out_offset = crop_idx * (3 * target_height * target_width);
    int hw = target_height * target_width;
    
    output[out_offset + 0 * hw + out_y * target_width + out_x] = R;
    output[out_offset + 1 * hw + out_y * target_width + out_x] = G;
    output[out_offset + 2 * hw + out_y * target_width + out_x] = B;
}

extern "C" {

/**
 * Convert RGB NvBufSurface (device) to float CHW tensor on device.
 * No GPU->CPU copy; use with infer_device for FPS optimization.
 */
int rgb_surface_to_tensor_device(
    NvBufSurface* rgb_surface,
    float* d_output_tensor,
    unsigned int num_images,
    unsigned int width,
    unsigned int height,
    int use_siglip_normalization
) {
    if (!rgb_surface || !d_output_tensor || num_images == 0)
        return -1;
    if (cudaSetDevice(rgb_surface->gpuId) != cudaSuccess)
        return -1;
    if (NvBufSurfaceMap(rgb_surface, -1, -1, NVBUF_MAP_READ) != 0)
        return -1;
    NvBufSurfaceSyncForDevice(rgb_surface, -1, -1);

    /* Single batched kernel launch (reduces launch overhead vs num_images launches) */
    SurfaceParams* d_params = nullptr;
    if (cudaMalloc(&d_params, num_images * sizeof(SurfaceParams)) != cudaSuccess) {
        NvBufSurfaceUnMap(rgb_surface, -1, -1);
        return -1;
    }
    SurfaceParams* h_params = (SurfaceParams*)malloc(num_images * sizeof(SurfaceParams));
    if (!h_params) {
        cudaFree(d_params);
        NvBufSurfaceUnMap(rgb_surface, -1, -1);
        return -1;
    }
    for (unsigned int i = 0; i < num_images; i++) {
        NvBufSurfaceParams* p = &rgb_surface->surfaceList[i];
        h_params[i].data = (unsigned char*)p->dataPtr;
        h_params[i].pitch = (int)p->pitch;
    }
    cudaMemcpy(d_params, h_params, num_images * sizeof(SurfaceParams), cudaMemcpyHostToDevice);
    free(h_params);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16, num_images);
    rgb_surface_batch_to_chw_float_kernel<<<grid, block>>>(
        d_params, (int)num_images, (int)width, (int)height, use_siglip_normalization ? 1 : 0, d_output_tensor);

    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
        err = cudaDeviceSynchronize();
    cudaFree(d_params);
    NvBufSurfaceUnMap(rgb_surface, -1, -1);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Convert RGBA NvBufSurface (device) to float CHW tensor on device.
 * No GPU->CPU copy; use with infer_device for FPS optimization.
 * Similar to rgb_surface_to_tensor_device but handles RGBA (4 bytes per pixel).
 */
int rgba_surface_to_tensor_device(
    NvBufSurface* rgba_surface,
    float* d_output_tensor,
    unsigned int num_images,
    unsigned int width,
    unsigned int height,
    int use_siglip_normalization
) {
    if (!rgba_surface || !d_output_tensor || num_images == 0)
        return -1;
    if (cudaSetDevice(rgba_surface->gpuId) != cudaSuccess)
        return -1;
    if (NvBufSurfaceMap(rgba_surface, -1, -1, NVBUF_MAP_READ) != 0)
        return -1;
    NvBufSurfaceSyncForDevice(rgba_surface, -1, -1);

    /* Single batched kernel launch (reduces launch overhead vs num_images launches) */
    SurfaceParams* d_params = nullptr;
    if (cudaMalloc(&d_params, num_images * sizeof(SurfaceParams)) != cudaSuccess) {
        NvBufSurfaceUnMap(rgba_surface, -1, -1);
        return -1;
    }
    SurfaceParams* h_params = (SurfaceParams*)malloc(num_images * sizeof(SurfaceParams));
    if (!h_params) {
        cudaFree(d_params);
        NvBufSurfaceUnMap(rgba_surface, -1, -1);
        return -1;
    }
    for (unsigned int i = 0; i < num_images; i++) {
        NvBufSurfaceParams* p = &rgba_surface->surfaceList[i];
        h_params[i].data = (unsigned char*)p->dataPtr;
        h_params[i].pitch = (int)p->pitch;
    }
    cudaMemcpy(d_params, h_params, num_images * sizeof(SurfaceParams), cudaMemcpyHostToDevice);
    free(h_params);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16, num_images);
    rgba_surface_batch_to_chw_float_kernel<<<grid, block>>>(
        d_params, (int)num_images, (int)width, (int)height, use_siglip_normalization ? 1 : 0, d_output_tensor);

    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
        err = cudaDeviceSynchronize();
    cudaFree(d_params);
    NvBufSurfaceUnMap(rgba_surface, -1, -1);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Preprocess multiple crops from an RGBA surface
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
) {
    if (!rgba_surface || !crop_boxes || !output || num_crops <= 0) {
        return -1;
    }
    
    // Copy crop boxes to device
    float* d_crop_boxes = nullptr;
    cudaMalloc(&d_crop_boxes, num_crops * 4 * sizeof(float));
    cudaMemcpyAsync(d_crop_boxes, crop_boxes, num_crops * 4 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    
    // Launch kernel
    dim3 block(16, 16);  // 16x16 threads per block
    dim3 grid(
        (target_width + block.x - 1) / block.x,
        (target_height + block.y - 1) / block.y,
        num_crops
    );
    
    preprocess_rgba_crops_kernel<<<grid, block, 0, stream>>>(
        (const unsigned char*)rgba_surface,
        surface_width,
        surface_height,
        surface_pitch,
        d_crop_boxes,
        num_crops,
        output,
        target_width,
        target_height
    );
    
    // Synchronize and check for errors
    cudaError_t err = cudaStreamSynchronize(stream);
    
    cudaFree(d_crop_boxes);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "[PREPROCESS] CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

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
) {
    if (!nv12_surface || !crop_boxes || !output || num_crops <= 0) {
        return -1;
    }
    
    // Copy crop boxes to device
    float* d_crop_boxes = nullptr;
    cudaMalloc(&d_crop_boxes, num_crops * 4 * sizeof(float));
    cudaMemcpyAsync(d_crop_boxes, crop_boxes, num_crops * 4 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    
    // Launch kernel
    dim3 block(16, 16);  // 16x16 threads per block
    dim3 grid(
        (target_width + block.x - 1) / block.x,
        (target_height + block.y - 1) / block.y,
        num_crops
    );
    
    preprocess_nv12_crops_kernel<<<grid, block, 0, stream>>>(
        (const unsigned char*)nv12_surface,
        surface_width,
        surface_height,
        surface_pitch,
        d_crop_boxes,
        num_crops,
        output,
        target_width,
        target_height
    );
    
    // Synchronize and check for errors
    cudaError_t err = cudaStreamSynchronize(stream);
    
    cudaFree(d_crop_boxes);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "[PREPROCESS] CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

/**
 * Map and get device pointer from NvBufSurface
 * Note: Caller must unmap the surface after use
 */
void* get_nvbufsurface_device_ptr(NvBufSurface* surf, unsigned int buf_idx) {
    if (!surf || buf_idx >= surf->numFilled) {
        fprintf(stderr, "[PREPROCESS] Invalid surface or buffer index\n");
        return nullptr;
    }
    
    // Map the surface to GPU memory for NVMM surfaces
    if (NvBufSurfaceMap(surf, buf_idx, -1, NVBUF_MAP_READ) != 0) {
        fprintf(stderr, "[PREPROCESS] Failed to map NvBufSurface\n");
        return nullptr;
    }
    
    // Sync to ensure data is available on device
    NvBufSurfaceSyncForDevice(surf, buf_idx, -1);
    
    NvBufSurfaceParams* params = &surf->surfaceList[buf_idx];
    return params->dataPtr;
}

/**
 * Unmap NvBufSurface after use
 */
void unmap_nvbufsurface(NvBufSurface* surf, unsigned int buf_idx) {
    if (surf && buf_idx < surf->numFilled) {
        NvBufSurfaceUnMap(surf, buf_idx, -1);
    }
}

} // extern "C"
