/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __GSTNVDSTENSOR_H__
#define __GSTNVDSTENSOR_H__

#include <cuda_runtime_api.h>
#include <gst/gst.h>
#include <vector>
#include "nvdstensor.h"

/**
 * This file describes the custom memory allocator for the Gstreamer TensorRT
 * plugin. The allocator allocates memory for a specified batch_size of frames
 * of resolution equal to the network input resolution.
 * The frames are allocated on device memory.
 */

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Holds the pointer for the allocated memory.
 */
typedef struct
{
  NvDsTensorArray *tensors;
  /** Vector of pointer to individual tensor memories in the batch memory */
  std::vector<void *> tensor_memory_ptrs;
} GstNvDsTensorMemory;

typedef struct _GstNvDsTensorAllocatorParams
{
  /** @param batch Batch of the network input. */
  guint batch;
  /** @param width Width of the network input. */
  guint width;
  /** @param height Height of the network input. */
  guint height;
  /** @param channels Channels of the network input. */
  guint channels;
  /** @param array_size Max size of batch that will be inferred. */
  guint array_size;

  /** @param tensor_format Tensor format of the buffers in the pool. */
  NvDsTensorFormat format;
  /** @param precision Precision of input data */
  NvDsTensorPrecision precision;

  /** @param gpu_id ID of the gpu where the batch memory will be allocated. */
  guint gpu_id;
} GstNvDsTensorAllocatorParams;

/**
 * Get GstNvDsTensorMemory structure associated with buffer allocated using
 * GstNvDsTensorAllocator.
 *
 * @param buffer GstBuffer allocated by this allocator.
 *
 * @return Pointer to the associated GstNvDsTensorMemory structure
 */
GstNvDsTensorMemory *gst_nvdstensor_buffer_get_memory (GstBuffer * buffer);

/**
 * Create a new GstNvDsTensorAllocator with the given parameters.
 *
 * @param params Tensor allocator params.
 *
 * @return Pointer to the GstNvDsTensorAllocator structure cast as GstAllocator
 */
GstAllocator *gst_nvdstensor_allocator_new (GstNvDsTensorAllocatorParams *params);

#if defined(__cplusplus)
}
#endif

#endif
