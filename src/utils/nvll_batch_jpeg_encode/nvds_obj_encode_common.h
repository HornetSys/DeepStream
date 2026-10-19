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

#ifndef __NVDS_ENCODE_OBJECT_COMMON_H__
#define __NVDS_ENCODE_OBJECT_COMMON_H__

#ifdef __cplusplus
extern "C"
{
#endif

#define get_max(x, y) (((x) > (y)) ? (x) : (y))
#define ROUND_DOWN_2(num)  ((num)&(~1))

typedef GList NvDsBufferList;

typedef struct _NvDsObjEncCtx
{
  NvDsBufferList *empty_list;
  int num_empty_elements;
  NvDsBufferList *full_list;
  int num_full_elements;
  GMutex empty_list_mutex;
  GCond buffer_cond;
  GCond frame_finish_cond;
  int num_cores;
  int is_integrated;
  NvBufSurface **scratch_buffer;
  gboolean buffers_alloted;
  GThreadPool *thread_pool;
  int gpu_id;
  cudaStream_t cuda_stream;
  int supports_unified_mem;
  void *handle_nvml_lib;
  void *handle_jpeg_lib;
  /* List of available & initialised encoder instances.
   * As allocating jtegra surfaces take up a lot of time,
   * maintaining two lists helps in re-using encoder instances
   * with already allocated surfaces.
   */
  GList *free_init_list;
  /* List of available & un-initialised encoder instances */
  GList *free_list;
  /* List of encode instances in-use */
  GList *occupied_list;
  GMutex enc_list_mutex;
  GCond signal_encode;
  /* To calculate avg encode time (in ms) */
  long double avg_encode_time;
  /* No. of encodes for which calcEncodeTime is enabled by the user */
  long long int total_encodes;
  long double total_encode_time;
} NvDsObjEncCtx;

typedef struct _NvDsJpegConvArgs
{
  NvDsBufferList *tempList;
  NvDsObjEncCtxHandle ctx;
  NvBufSurface *scratchBuffer;
  NvDsObjectMeta *objMeta;
  NvDsFrameMeta *frameMeta;
  NvDsObjEncUsrArgs *userData;
  uint width;
  uint height;
  long long int initTimeSec;
  long long int initTimeUsec;
} NvDsJpegConvArgs;

void NvDsCloseJpegLib (void *);

void NvDsCloseJpegLib_DGPU (void *);

void NvDsOpenJpegLib (void *);

void NvDsOpenJpegLib_DGPU (void *);

gboolean jpeg_convert (NvDsJpegConvArgs *);

gboolean jpeg_convert_dgpu (NvDsJpegConvArgs *);

gboolean allocate_buffers (NvDsObjEncCtxHandle, NvBufSurface *, int, int);

gboolean deallocate_buffers (NvDsObjEncCtxHandle);

void jpeg_write_to_file (uint8_t *, uint64_t, char *);

void calculate_encode_time (NvDsJpegConvArgs *);

gpointer nvds_copy_image_encode_meta (gpointer, gpointer);

void nvds_release_image_encode_meta (gpointer, gpointer);

NvDsBufferList *nvds_acquire_buffer_from_pool (NvDsObjEncCtxHandle);

void nvds_release_buffer_to_pool (NvDsObjEncCtxHandle, NvDsBufferList *);

/* Initialises encoder instances by allocating memory for structures and
 * calling jpeg function calls.
 */
void init_encoder_instances (NvDsObjEncCtx *);
void init_encoder_instances_dgpu (NvDsObjEncCtx *);

/* Frees allocated structures and calls destroy on encoder instances */
void destroy_encoder_instances (NvDsObjEncCtxHandle);
void destroy_encoder_instances_dgpu (NvDsObjEncCtxHandle);

/* Acquires an encoder instance from the pool of encoder instances.
 * Acquire takes place from the free_init_list first for an
 * encoder instance whose jtegra surfaces have already been allocated.
 * If no such instances are available then acquire takes place from the
 * free_list which has encoder instances whose jtegra surfaces haven't been
 * allocated.
 */
GList *acquire_encoder_instance (NvDsObjEncCtxHandle);
GList *acquire_encoder_instance_dgpu (NvDsObjEncCtxHandle);

/* Releases the encoder instance back to the free_init_list after encode
 * has finished.
 */
void release_encoder_instance (NvDsObjEncCtxHandle, GList *);
void release_encoder_instance_dgpu (NvDsObjEncCtxHandle, GList *);


#ifdef __cplusplus
}
#endif

#endif
