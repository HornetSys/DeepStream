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

#include "gst-nvcommon.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

GType
gst_nvbuf_memory_get_type (void)
{
  static gsize memory_format_type = 0;
  static const GEnumValue memory_format[] = {
    { NVBUF_MEM_DEFAULT,
      "Default memory allocated, specific to particular platform", "nvbuf-mem-default"},

    { NVBUF_MEM_CUDA_PINNED,
      "Allocate Pinned/Host cuda memory", "nvbuf-mem-cuda-pinned"},
    { NVBUF_MEM_CUDA_DEVICE,
      "Allocate Device cuda memory", "nvbuf-mem-cuda-device"},
    { NVBUF_MEM_CUDA_UNIFIED,
      "Allocate Unified cuda memory", "nvbuf-mem-cuda-unified"},
#ifdef __aarch64__
    { NVBUF_MEM_SURFACE_ARRAY,
      "Allocate Surface Array memory, applicable for Jetson", "nvbuf-mem-surface-array"},
/*    { NVBUF_MEM_HANDLE,
      "Allocate Surface Handle memory, applicable for Jetson", "nvbuf-mem-handle"},*/
#endif
    /*
    { NVBUF_MEM_SYSTEM,
      "Allocate Surface System memory, allocated using calloc", "nvbuf-mem-system"},*/
    {0, NULL, NULL}
  };

  if (g_once_init_enter (&memory_format_type)) {
    GType tmp = g_enum_register_static ("GstNvBufMemoryType",
        memory_format);
    g_once_init_leave (&memory_format_type, tmp);
  }

  return (GType) memory_format_type;
}

static const char * const NvBufSurfaceMemType_names[] =
{
  [NVBUF_MEM_DEFAULT] = "nvbuf-mem-default",
  [NVBUF_MEM_CUDA_PINNED] = "nvbuf-mem-cuda-pinned",
  [NVBUF_MEM_CUDA_DEVICE] = "nvbuf-mem-cuda-device",
  [NVBUF_MEM_CUDA_UNIFIED] = "nvbuf-mem-cuda-unified",
  [NVBUF_MEM_SURFACE_ARRAY] = "nvbuf-mem-surface-array",
  [NVBUF_MEM_HANDLE] = "nvbuf-mem-handle",
  [NVBUF_MEM_SYSTEM] = "nvbuf-mem-system",
};

const char * gst_nvbuf_memory_get_name(int mem_type)
{
    return NvBufSurfaceMemType_names[(NvBufSurfaceMemType)mem_type];
}

int gst_nvbuf_memory_get_value(const char *mem_type_string)
{
    for(int i= NVBUF_MEM_DEFAULT; i <= NVBUF_MEM_SYSTEM; i++)
    {
        if(g_str_equal(mem_type_string, NvBufSurfaceMemType_names[(NvBufSurfaceMemType)i]))
        {
            return i;
        }
    }
    return -1;
}

//WINDOWS
#ifdef _WIN32
int CHECK_NVDS_MEMORY_AND_GPUID(GstElement* object, NvBufSurface* surface, int gpu_id)
{
    int _errtype = 0;
    do
    {
        if (((surface->memType == NVBUF_MEM_DEFAULT || surface->memType == NVBUF_MEM_CUDA_DEVICE) &&
            ((int)surface->gpuId != (int)gpu_id)) ||
            (((int)surface->gpuId == (int)gpu_id) && (surface->memType == NVBUF_MEM_SYSTEM)))
        {
            GST_ELEMENT_ERROR(object, RESOURCE, FAILED,
                ("Memory Compatibility Error:Input surface gpu-id doesnt match with configured gpu-id for element,"
                    " please allocate input using unified memory, or use same gpu-ids OR,"
                    " if same gpu-ids are used ensure appropriate Cuda memories are used"),
                ("surface-gpu-id=%d,%s-gpu-id=%d,surface->memType:%d", surface->gpuId, GST_ELEMENT_NAME(object),
                    gpu_id, surface->memType));
            _errtype = 1;
        }
    } while (0);
    return _errtype;
}
#endif

GType
gst_compute_hw_get_type (void)
{
  static gsize compute_hw_type = 0;
  static const GEnumValue compute_hw[] = {
    {NvBufSurfTransformCompute_Default, "Default, GPU for Tesla, VIC for Jetson", "Default"},
    {NvBufSurfTransformCompute_GPU, "GPU", "GPU"},
    {NvBufSurfTransformCompute_VIC, "VIC", "VIC"},
    {0, NULL, NULL},
  };

  if(g_once_init_enter (&compute_hw_type)) {
    GType tmp = g_enum_register_static ("GstNvComputeHWType",
       compute_hw);
    g_once_init_leave(&compute_hw_type, tmp);
  }

  return (GType) compute_hw_type;
}

GType
gst_video_interpolation_method_get_type (void)
{
  static gsize interpolation_method_type = 0;
  static const GEnumValue video_interpolation_methods[] = {
    {NvBufSurfTransformInter_Nearest, "Nearest", "Nearest"},
    {NvBufSurfTransformInter_Bilinear, "Bilinear", "Bilinear"},
    {NvBufSurfTransformInter_Algo1, "GPU - Cubic, VIC - 5 Tap", "Algo-1"},
    {NvBufSurfTransformInter_Algo2, "GPU - Super, VIC - 10 Tap", "Algo-2"},
    {NvBufSurfTransformInter_Algo3, "GPU - LanzoS, VIC - Smart", "Algo-3"},
    {NvBufSurfTransformInter_Algo4, "GPU - Ignored, VIC - Nicest", "Algo-4"},
    {NvBufSurfTransformInter_Default, "GPU - Nearest, VIC - Nearest", "Default"},
    {0, NULL, NULL},
  };

  if(g_once_init_enter (&interpolation_method_type)) {
    GType tmp = g_enum_register_static ("GstNvInterpolationMethod",
      video_interpolation_methods);
    g_once_init_leave(&interpolation_method_type, tmp);
  }

  return (GType) interpolation_method_type;
}


