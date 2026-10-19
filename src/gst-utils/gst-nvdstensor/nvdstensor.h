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

/**
 * @file nvdstensor.h
 * <b>NvDsTensor Interface </b>
 *
 * This file specifies the NvDsTensor management API.
 *
 * The NvDsTensor API provides methods to allocate / deallocate, map / unmap
 * and copy batched buffers.
 */

#ifndef NVDSTENSOR_H_
#define NVDSTENSOR_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** @name NvDsTensor types and functions.
 * This section describes types and functions of NvDsTensor application
 * programming interface.
 */

/** @{ */

/** Defines the default padding length for reserved fields of structures. */
#define STRUCTURE_PADDING  4

/**
 * Specifies mapping types for \ref NvDsTensor.
 */
typedef enum
{
  NVDSTENSOR_MAP_READ,       /**< Specifies \ref NvDsTensor mapping type "read." */
  NVDSTENSOR_MAP_WRITE,      /**< Specifies \ref NvDsTensor mapping type "write." */
  NVDSTENSOR_MAP_READ_WRITE, /**< Specifies \ref NvDsTensor mapping type "read/write." */
} NvDsTensorMemMapFlags;

typedef enum
{
  NVDSTENSOR_NCHW,
} NvDsTensorFormat;

/**
 * Specifies memory types for \ref NvDsTensor.
 */
typedef enum
{
  NVDSTENSOR_MEM_DEFAULT,
  /** Specifies CUDA Host memory type. */
  NVDSTENSOR_MEM_CUDA_PINNED,
  /** Specifies CUDA Device memory type. */
  NVDSTENSOR_MEM_CUDA_DEVICE,
  /** Specifies CUDA Unified memory type. */
  NVDSTENSOR_MEM_CUDA_UNIFIED,
  /** Specifies memory allocated by malloc(). */
  NVDSTENSOR_MEM_SYSTEM,
} NvDsTensorMemType;

typedef enum
{
  NVDSTENSOR_FP32,
  NVDSTENSOR_FP16,
  NVDSTENSOR_INT32,
  NVDSTENSOR_INT16,
  NVDSTENSOR_INT8
} NvDsTensorPrecision;

/**
 * Holds parameters required to allocate an \ref NvDsTensor
 */
typedef struct NvDsTensorCreateParams {
  /** Holds the GPU ID. Valid only for a multi-GPU system. */
  uint32_t gpuId;

  uint32_t n;
  uint32_t c;
  uint32_t h;
  uint32_t w;

  /** Holds the amount of memory to be allocated. Optional; if set, all other
   parameters (width, height, etc.) are ignored. */
  uint32_t size;

  /** Holds a "contiguous memory" flag. If set, contiguous memory is allocated
   for the batch. Valid only for CUDA memory types. */
  bool isContiguous;

  /** Holds the type of memory to be allocated. */
  NvDsTensorMemType memType;
  NvDsTensorFormat format;
  NvDsTensorPrecision precision;

} NvDsTensorCreateParams;

/**
 * Holds information about a single buffer in a batch.
 */
typedef struct NvDsTensor {
  uint32_t n;
  uint32_t c;
  uint32_t w;
  uint32_t h;

  /** Holds a pointer to allocated memory. */
  void * dataPtr;
  /** Holds pointers to mapped buffers. Initialized to NULL
   when the structure is created. */
  void * mappedAddr;

  /** Holds the amount of allocated memory. */
  uint32_t dataSize;

  void * _reserved[STRUCTURE_PADDING];
} NvDsTensor;

/**
 * Holds information about batched buffers.
 */
typedef struct NvDsTensorArray {
  /** Holds a GPU ID. Valid only for a multi-GPU system. */
  uint32_t gpuId;

  /** Holds an "is contiguous" flag. If set, memory allocated for the batch
   is contiguous. */
  bool isContiguous;

  /** Holds type of memory for buffers in the batch. */
  NvDsTensorMemType memType;
  /** Holds precision for buffers in the batch. */
  NvDsTensorPrecision precision;
  /** Holds the tensor format for buffers in the batch. */
  NvDsTensorFormat format;

  /** Holds the batch size. */
  uint32_t arraySize;
  /** Holds the number valid and filled buffers. Initialized to zero when
   an instance of the structure is created. */
  uint32_t numFilled;
  /** Holds a pointer to an array of batched buffers. */
  NvDsTensor *tensorArray;

  void * _reserved[STRUCTURE_PADDING];
} NvDsTensorArray;

/**
 * \brief  Allocates a batch of tensor.
 *
 * Allocates memory for \a batchSize tensors and returns a pointer to an
 * allocated \ref NvDsTensorArray. The \a params structure must have
 * the allocation parameters of a single buffer. If \a params.size
 * is set, a buffer of that size is allocated, and all other
 * parameters (width, height, tensor format, etc.) are ignored.
 *
 * Call NvDsTensorDestroy() to free resources allocated by this function.
 *
 * @param[out] tensor array An indirect pointer to the allocated batched
 *                           buffers.
 * @param[in]  arraySize    Batch size of buffers.
 * @param[in]  params       A pointer to an \ref NvDsTensorCreateParams
 *                           structure.
 *
 * @return 0 if successful, or -1 otherwise.
 */
int NvDsTensorCreate (NvDsTensorArray **tensor, uint32_t arraySize,
                                NvDsTensorCreateParams *params);

/**
 * \brief  Frees batched buffers previously allocated by NvDsTensorCreate().
 *
 * @param[in] tensors  A pointer to an \ref NvDsTensorArray to be freed.
 *
 * @return 0 if successful, or -1 otherwise.
 */
int NvDsTensorDestroy (NvDsTensorArray *tensor);

/**
 * \brief  Maps hardware batched buffers to the HOST or CPU address space.
 *
 * This function fills an array of pointers at
 * \a tensors->tensorArray->mappedAddr
 * \a tensors is a pointer to an \ref NvDsTensorArray.
 * \a tensorArray is a pointer to an \ref NvDsTensor.
 * \a mappedAddr holds pointer to buffer.
 *
 * Use NvDsTensorUnMap() to unmap buffer(s) and release any resource.
 *
 * @param[in,out] tensors  A pointer to an NvDsTensorArray structure. The function
 *                         stores pointers to the buffers in a descendant of this
 *                         structure; see the notes above.
 * @param[in] index        Index of a buffer in the batch. -1 refers to all buffers
 *                         in the batch.
 * @param[in] type         A flag for mapping type.
 *
 * @return 0 if successful, or -1 otherwise.
 */
int NvDsTensorMap (NvDsTensorArray *tensors, int index, NvDsTensorMemMapFlags type);

/**
 * \brief  Unmaps previously mapped buffer(s).
 *
 * @param[in] tensors   A pointer to an \ref NvDsTensorArray structure.
 * @param[in] index     Index of a buffer in the batch. -1 indicates
 *                      all buffers in the batch.
 *
 * @return  0 if successful, or -1 otherwise.
 */
int NvDsTensorUnMap (NvDsTensorArray *tensors, int index);

/** @} */

#ifdef __cplusplus
}
#endif
#endif /* NVDSTENSOR_H_ */
