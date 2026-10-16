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

#include "nvdstensor.h"
#include "nvdstensor_common.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

static int NvDsTensorMapCudaBatch (NvDsTensorArray *tensors, int index,
                                             NvDsTensorMemMapFlags type)
{
  unsigned int i;

  for (i = 0; i < tensors->numFilled; i++) {
    NvDsTensor *pTensor = &tensors->tensorArray[i];

    pTensor->mappedAddr = pTensor->dataPtr;
  }

  return 0;
}

int NvDsTensorCreate (NvDsTensorArray **tensors, uint32_t arraySize,
                                      NvDsTensorCreateParams *params)
{
  int ret = -1;
  NvDsTensorArray *pTensors = NULL;

  if (!tensors || !params) {
    printf ("nvdstensor: NvDsTensorCreate() NULL arguments\n");
    return ret;
  }

  pTensors = (NvDsTensorArray *) calloc (1, sizeof(NvDsTensorArray));

  pTensors->tensorArray = (NvDsTensor *) calloc (arraySize, sizeof(NvDsTensor));
  pTensors->gpuId = params->gpuId;
  pTensors->memType = params->memType;

  switch (params->memType) {
    case NVDSTENSOR_MEM_DEFAULT:
    case NVDSTENSOR_MEM_CUDA_DEVICE:
    case NVDSTENSOR_MEM_CUDA_PINNED:
    case NVDSTENSOR_MEM_CUDA_UNIFIED:
      ret = CreateCudaBufferBatch (pTensors, arraySize, params);
      break;
    case NVDSTENSOR_MEM_SYSTEM:
      ret = CreateSystemBufferBatch (pTensors, arraySize, params);
      break;
    default:
      printf ("nvdstensor: memory type (%d) not supported\n", params->memType);
      return ret;
  }

  if (ret < 0) {
    if (pTensors->arraySize) {
      // release partially allocated batch.
      NvDsTensorDestroy (pTensors);
    } else {
      free (pTensors->tensorArray);
      free (pTensors);
    }
  } else {
    *tensors = pTensors;
  }
  return ret;
}

int NvDsTensorDestroy (NvDsTensorArray *tensors)
{
  int ret = -1;

  if(!tensors) {
    printf ("nvdstensor: invalid argument\n");
    return ret;
  }

  switch (tensors->memType) {
    case NVDSTENSOR_MEM_DEFAULT:
    case NVDSTENSOR_MEM_CUDA_DEVICE:
    case NVDSTENSOR_MEM_CUDA_PINNED:
    case NVDSTENSOR_MEM_CUDA_UNIFIED:
      ret = ReleaseCudaBufferBatch (tensors);
      if (ret)
        printf ("nvdstensor: Error(%d) in releasing cuda memory\n", ret);
      break;
    case NVDSTENSOR_MEM_SYSTEM:
      ret = ReleaseSystemBufferBatch (tensors);
      if (ret)
        printf ("nvdstensor: Error (%d) in releasing system memory\n", ret);
      break;
    default:
      printf ("nvdstensor: memory type (%d) not supported\n", tensors->memType);
      break;
  }

  free (tensors->tensorArray);
  free (tensors);
  return ret;
}

int NvDsTensorMap (NvDsTensorArray *tensors, int index, NvDsTensorMemMapFlags type)
{
  int ret = -1;

  if(!tensors) {
    printf ("nvdstensor: invalid argument\n");
    return ret;
  }

  switch (tensors->memType) {
    case NVDSTENSOR_MEM_CUDA_UNIFIED:
      ret = NvDsTensorMapCudaBatch (tensors, index, type);
      break;
    case NVDSTENSOR_MEM_DEFAULT:
    case NVDSTENSOR_MEM_CUDA_DEVICE:
    case NVDSTENSOR_MEM_CUDA_PINNED:
    default:
      printf ("nvdstensor: mapping of memory type (%d) not supported\n", tensors->memType);
      break;
  }
  return ret;
}

int NvDsTensorUnMap (NvDsTensorArray *tensors, int index)
{
  //TODO: should pointers be reseted?
  return 0;
}
