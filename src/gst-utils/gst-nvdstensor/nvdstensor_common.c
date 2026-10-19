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
#include <stdlib.h>
#include <string.h>
#include "nvdstensor_common.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        printf ("Cuda failure: status=%d\n", status);   \
        goto error;                                     \
    }                                                   \
}

int NvDsTensorIsCudaMem (NvDsTensorArray *tensors)
{
  if (!tensors)
    return 0;

  switch (tensors->memType) {
    case NVDSTENSOR_MEM_CUDA_DEVICE:
    case NVDSTENSOR_MEM_CUDA_PINNED:
    case NVDSTENSOR_MEM_CUDA_UNIFIED:
    case NVDSTENSOR_MEM_DEFAULT:
      return 1;
    default:
      return 0;
  }
}

int CreateCudaBufferBatch (NvDsTensorArray *tensors, uint32_t arraySize,
                           NvDsTensorCreateParams *params)
{
  int ret = -1;
  unsigned int i;
  int currentDevice = -1;
  NvDsTensor *pTensor;
  unsigned int size, count, memSize;

  if (!tensors || !params) {
    printf ("nvdstensor: bad parameters\n");
    return ret;
  }

  if (!arraySize) {
    printf ("nvdstensor: Wrong batch size(%d), Using one as default\n", arraySize);
    arraySize = 1;
  }

  if (!params->size && (!params->w || !params->h || !params->c)) {
    printf ("nvdstensor: bad parameters\n");
    return ret;
  }

  tensors->precision = params->precision;
  tensors->format = params->format;
  tensors->memType = params->memType;

  if (params->size) {
    size = params->size;
  } else {
    switch (params->precision) {
      case NVDSTENSOR_FP32:
        size = sizeof (float);
      break;
      case NVDSTENSOR_FP16:
        size = sizeof (int16_t);
      break;
      case NVDSTENSOR_INT32:
        size = sizeof (int32_t);
      break;
      case NVDSTENSOR_INT16:
        size = sizeof (int16_t);
      break;
      case NVDSTENSOR_INT8:
        size = sizeof (int8_t);
      break;
      default:
        size = 1;
      break;
    }
    size = size * params->c * params->h * params->w * params->n;
  }

  /* Changed to - 1 to catch failures in following calls */
  ret = -1;
  // store the current device of calling host thread.
  CHECK(cudaGetDevice(&currentDevice));
  if (currentDevice != (int) params->gpuId) {
    CHECK(cudaSetDevice(params->gpuId));
  }

  if (params->isContiguous) {
    count = 1;
    memSize = size * arraySize;
  } else {
    count = arraySize;
    memSize = size;
  }

  for (i = 0; i < count; i++) {
    pTensor = &tensors->tensorArray[i];
    memset (pTensor, 0 , sizeof(NvDsTensor));

    switch (params->memType) {
      case NVDSTENSOR_MEM_CUDA_PINNED:
        CHECK (cudaMallocHost (&pTensor->dataPtr, memSize));
        CHECK (cudaMemset(pTensor->dataPtr, 0x00, memSize));
        break;
      case NVDSTENSOR_MEM_DEFAULT:
      case NVDSTENSOR_MEM_CUDA_DEVICE:
        CHECK (cudaMalloc (&pTensor->dataPtr, memSize));
        CHECK (cudaMemset(pTensor->dataPtr, 0x00, memSize));
        break;
      case NVDSTENSOR_MEM_CUDA_UNIFIED:
        CHECK (cudaMallocManaged (&pTensor->dataPtr, memSize, 1));
        CHECK (cudaMemset(pTensor->dataPtr, 0x00, memSize));
        break;
      default:
        break;
    }

    tensors->arraySize++;
  }

  for (i = 0; i < arraySize; i++) {
    pTensor = &tensors->tensorArray[i];

    pTensor->n = params->n;
    pTensor->c = params->c;
    pTensor->h = params->h;
    pTensor->w = params->w;
    pTensor->dataSize = size;

    /* For contiguous memory, use same buffer descriptor for all buffers
     * and adjust offset of buffers accordingly.
     */
    if (params->isContiguous) {
      pTensor->dataPtr = (unsigned char *) tensors->tensorArray[0].dataPtr + i * size;
    }
  }

  tensors->numFilled = 0;
  tensors->arraySize = arraySize;
  tensors->isContiguous = params->isContiguous;
  ret = 0;

error:
  // restore the device of calling host thread.
  if (currentDevice != -1 && currentDevice != (int) params->gpuId) {
    cudaSetDevice(currentDevice);
  }
  return ret;
}

int ReleaseCudaBufferBatch (NvDsTensorArray *tensors)
{
  int ret = -1;
  NvDsTensor *pTensor;
  int currentDevice = -1;
  unsigned int i, count;

  if (!tensors) {
    printf ("nvdstensor: Invalid argument\n");
    return ret;
  }

  // store the current device of calling host thread.
  CHECK(cudaGetDevice(&currentDevice));
  if (currentDevice != (int) tensors->gpuId) {
    CHECK(cudaSetDevice(tensors->gpuId));
  }

  if (tensors->arraySize && tensors->tensorArray[0].dataPtr) {
    if (tensors->isContiguous) {
      count = 1;
    } else {
      count = tensors->arraySize;
    }

    for (i = 0; i < count; i++) {
      pTensor = &tensors->tensorArray[i];

      switch (tensors->memType) {
        case NVDSTENSOR_MEM_CUDA_PINNED:
          if (pTensor->dataPtr) {
            CHECK (cudaFreeHost (pTensor->dataPtr));
          }
          ret = 0;
          break;
        case NVDSTENSOR_MEM_DEFAULT:
        case NVDSTENSOR_MEM_CUDA_DEVICE:
          if (pTensor->dataPtr) {
            CHECK (cudaFree (pTensor->dataPtr));
          }
          ret = 0;
          break;
        case NVDSTENSOR_MEM_CUDA_UNIFIED:
          if (pTensor->dataPtr) {
            CHECK (cudaFree (pTensor->dataPtr));
          }
          ret = 0;
          break;
        default:
          printf ("nvdstensor: invalid memory type (%d)\n", tensors->memType);
          break;
      }
      pTensor->dataPtr = NULL;
    }
  }

error:
  // restore the device of calling host thread.
  if (currentDevice != -1 && currentDevice != (int) tensors->gpuId) {
    cudaSetDevice(currentDevice);
  }
  return ret;
}

int CreateSystemBufferBatch (NvDsTensorArray *tensors, uint32_t arraySize,
                             NvDsTensorCreateParams *params)
{
  int ret = -1;
  unsigned int i;
  NvDsTensor *pTensor;
  unsigned int size;

  if (!tensors || !params) {
    printf ("nvdstensor: bad parameters\n");
    return ret;
  }

  if (!arraySize) {
    printf ("nvdstensor: bad parameters\n");
    return ret;
  }

  if (!params->size && (!params->w || !params->h || !params->c)) {
    printf ("nvdstensor: bad parameters\n");
    return ret;
  }

  tensors->precision = params->precision;
  tensors->format = params->format;
  tensors->memType = params->memType;

  if (params->size) {
    size = params->size;
  } else {
    size = params->c * params->h * params->w * params->n; //add precision to params and tensor
  }

  /* Changed to - 1 to catch failures in following calls */
  ret = -1;
  for (i = 0; i < arraySize; i++) {
    pTensor = &tensors->tensorArray[i];
    memset (pTensor, 0 , sizeof(NvDsTensor));

    pTensor->n = params->n;
    pTensor->c = params->c;
    pTensor->h = params->h;
    pTensor->w = params->w;
    pTensor->dataSize = size;

    pTensor->dataPtr = calloc (1, size);

    if (!pTensor->dataPtr) {
      printf ("nvdstensor: failed to allocate memory type (%d)\n", params->memType);
      return ret;
    }

    tensors->arraySize++;
  }

  tensors->numFilled = 0;
  return 0;
}

int ReleaseSystemBufferBatch (NvDsTensorArray *tensors)
{
  unsigned int i;
  NvDsTensor *pTensor;

  if (!tensors) {
    printf ("nvdstensor: Invalid argument\n");
    return -1;
  }

  for (i = 0; i < tensors->arraySize; i++) {
    pTensor = &tensors->tensorArray[i];

    free (pTensor->dataPtr);
    pTensor->dataPtr = NULL;
  }
  return 0;
}
