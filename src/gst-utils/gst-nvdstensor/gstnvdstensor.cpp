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

#include <stdio.h>
#include "cuda_runtime.h"
#include "gstnvdstensor.h"

/* Standard GStreamer boiler plate macros */
#define GST_TYPE_NVDSTENSOR_ALLOCATOR \
    (gst_nvdstensor_allocator_get_type ())
#define GST_NVDSTENSOR_ALLOCATOR(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSTENSOR_ALLOCATOR,GstNvDsTensorAllocator))
#define GST_NVDSTENSOR_ALLOCATOR_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSTENSOR_ALLOCATOR,GstNvDsTensorAllocatorClass))
#define GST_IS_NVDSTENSOR_ALLOCATOR(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSTENSOR_ALLOCATOR))
#define GST_IS_NVDSTENSOR_ALLOCATOR_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSTENSOR_ALLOCATOR))

typedef struct _GstNvDsTensorAllocator GstNvDsTensorAllocator;
typedef struct _GstNvDsTensorAllocatorClass GstNvDsTensorAllocatorClass;

G_GNUC_INTERNAL GType gst_nvdstensor_allocator_get_type (void);

GST_DEBUG_CATEGORY_STATIC (gst_nvdstensor_allocator_debug);
#define GST_CAT_DEFAULT gst_nvdstensor_allocator_debug

/**
 * Extends the GstAllocator class. Holds the parameters for allocator.
 */
struct _GstNvDsTensorAllocator
{
  GstAllocator allocator;

  guint array_size;

  guint n;
  guint c;
  guint h;
  guint w;

  guint gpu_id;

  NvDsTensorFormat format;
  NvDsTensorPrecision precision;
};

struct _GstNvDsTensorAllocatorClass
{
  GstAllocatorClass parent_class;
};

/* Standard boiler plate to create a debug category and initializing the
 * allocator type.
 */
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvdstensor_allocator_debug, "nvdstensorallocator", 0, "nvdstensor allocator");
#define gst_nvdstensor_allocator_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvDsTensorAllocator, gst_nvdstensor_allocator,
    GST_TYPE_ALLOCATOR, _do_init);

/** Type of the memory allocated by the allocator. This can be used to identify
 * buffers / memories allocated by this allocator. */
#define GST_NVDSTENSOR_MEMORY_TYPE "nvdstensor"

/** Structure allocated internally by the allocator. */
typedef struct
{
  /** Should be the first member of a structure extending GstMemory. */
  GstMemory mem;
  GstNvDsTensorMemory tensor_mem;
} GstNvDsTensorMem;

/* Function called by GStreamer buffer pool to allocate memory using this
 * allocator. */
static GstMemory *
gst_nvdstensor_allocator_alloc (GstAllocator * allocator, gsize size,
    GstAllocationParams * params)
{
  GstNvDsTensorAllocator *tensor_allocator = GST_NVDSTENSOR_ALLOCATOR (allocator);
  GstNvDsTensorMem *nvmem = new GstNvDsTensorMem;
  GstNvDsTensorMemory *tmem = &nvmem->tensor_mem;
  NvDsTensorCreateParams create_params = { 0 };

  if (!tensor_allocator) {
    GST_ERROR ("Error: Could not nvdstensor allocator");
    delete nvmem;
    return nullptr;
  }

  create_params.gpuId = tensor_allocator->gpu_id;

  create_params.n = tensor_allocator->n;
  create_params.c = tensor_allocator->c;
  create_params.h = tensor_allocator->h;
  create_params.w = tensor_allocator->w;

  create_params.size = 0;
  create_params.isContiguous = 1;
  create_params.format = tensor_allocator->format;
  create_params.memType = NVDSTENSOR_MEM_DEFAULT;

  if (NvDsTensorCreate (&tmem->tensors, tensor_allocator->array_size,
          &create_params) != 0) {
    GST_ERROR ("Error: Could not allocate internal buffer pool for nvdstensor");
    delete nvmem;
    return nullptr;
  }

  tmem->tensor_memory_ptrs.assign (tensor_allocator->array_size, nullptr);

  for (guint i = 0; i < tensor_allocator->array_size; i++) {
    /* Calculate pointers to individual frame memories in the batch memory and
     * insert in the vector. */
    tmem->tensor_memory_ptrs[i] = (char *) tmem->tensors->tensorArray[i].dataPtr;
  }

  /* Initialize the GStreamer memory structure. */
  gst_memory_init ((GstMemory *) nvmem, (GstMemoryFlags) 0, allocator, nullptr,
      sizeof (GstNvDsTensorMemory), params->align, 0, size);

  return (GstMemory *) nvmem;
}

/* Function called by buffer pool for freeing memory using this allocator. */
static void
gst_nvdstensor_allocator_free (GstAllocator * allocator, GstMemory * memory)
{
  GstNvDsTensorMem *nvmem = (GstNvDsTensorMem *) memory;
  GstNvDsTensorMemory *tmem = &nvmem->tensor_mem;

  NvDsTensorDestroy (tmem->tensors);

  delete nvmem;
}

/* Function called when mapping memory allocated by this allocator. Should
 * return pointer to GstNvDsTensorMemory. */
static gpointer
gst_nvdstensor_memory_map (GstMemory * mem, gsize maxsize, GstMapFlags flags)
{
  GstNvDsTensorMem *nvmem = (GstNvDsTensorMem *) mem;

  return (gpointer) & nvmem->tensor_mem;
}

static void
gst_nvdstensor_memory_unmap (GstMemory * mem)
{
}

/* Standard boiler plate. Assigning implemented function pointers. */
static void
gst_nvdstensor_allocator_class_init (GstNvDsTensorAllocatorClass * klass)
{
  GstAllocatorClass *allocator_class = GST_ALLOCATOR_CLASS (klass);

  allocator_class->alloc = GST_DEBUG_FUNCPTR (gst_nvdstensor_allocator_alloc);
  allocator_class->free = GST_DEBUG_FUNCPTR (gst_nvdstensor_allocator_free);
}

/* Standard boiler plate. Assigning implemented function pointers and setting
 * the memory type. */
static void
gst_nvdstensor_allocator_init (GstNvDsTensorAllocator * allocator)
{
  GstAllocator *parent = GST_ALLOCATOR_CAST (allocator);

  parent->mem_type = GST_NVDSTENSOR_MEMORY_TYPE;
  parent->mem_map = gst_nvdstensor_memory_map;
  parent->mem_unmap = gst_nvdstensor_memory_unmap;
}

/* Create a new allocator of type GST_TYPE_NVDSTENSOR_ALLOCATOR and initialize
 * members. */
GstAllocator *
gst_nvdstensor_allocator_new (GstNvDsTensorAllocatorParams *params)
{
  GstNvDsTensorAllocator *allocator = (GstNvDsTensorAllocator *)
      g_object_new (GST_TYPE_NVDSTENSOR_ALLOCATOR,
      nullptr);

  allocator->n = params->batch;
  allocator->c = params->channels;
  allocator->h = params->height;
  allocator->w = params->width;

  allocator->array_size = params->array_size;

  allocator->gpu_id = params->gpu_id;
  allocator->format = params->format;
  allocator->precision = params->precision;

  return (GstAllocator *) allocator;
}

GstNvDsTensorMemory *
gst_nvdstensor_buffer_get_memory (GstBuffer * buffer)
{
  GstMemory *mem;

  mem = gst_buffer_peek_memory (buffer, 0);

  if (!mem || !gst_memory_is_type (mem, GST_NVDSTENSOR_MEMORY_TYPE))
    return nullptr;

  return &(((GstNvDsTensorMem *) mem)->tensor_mem);
}
