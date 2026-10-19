/**
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

#include <glib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvdsmeta.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvds_obj_encode.h"
#include "nvds_obj_encode_common.h"
#include "nvml.h"

static typeof(&nvmlInit) nvmlInit_fn;
static typeof(&nvmlDeviceGetHandleByIndex) nvmlDeviceGetHandleByIndex_fn;
static typeof(&nvmlDeviceGetBrand) nvmlDeviceGetBrand_fn;
static typeof(&nvmlShutdown) nvmlShutdown_fn;

void nvds_obj_enc_finish (NvDsObjEncCtxHandle ctx);

/* Write to File */
void
jpeg_write_to_file (uint8_t * out_buffer, uint64_t out_len, char *filename_buf)
{
  FILE *file;
  file = fopen (filename_buf, "wb");
  fwrite (out_buffer, sizeof (uint8_t), out_len, file);
  fclose (file);
}

void
calculate_encode_time (NvDsJpegConvArgs * conv_args)
{
  struct timeval t1;
  gettimeofday(&t1, NULL);
  long long int diff_sec = t1.tv_sec - conv_args->initTimeSec;
  long long int diff_usec = t1.tv_usec - conv_args->initTimeUsec;
  if (diff_usec < 0) {
    --diff_sec;
    diff_usec += 1000000;
  }
  long double diff_msec = diff_sec*1000 + (1.0 * diff_usec)/1000;
  printf("Time of encode for %s - %Lf ms\n", conv_args->userData->fileNameImg, diff_msec);
  conv_args->ctx->total_encode_time += diff_msec;
  conv_args->ctx->total_encodes++;
}

/* copy function set by user. "data" holds a pointer to NvDsUserMeta */
gpointer
nvds_copy_image_encode_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsObjEncOutParams *src_user_metadata =
      (NvDsObjEncOutParams *) user_meta->user_meta_data;
  NvDsObjEncOutParams *dst_user_metadata =
      (NvDsObjEncOutParams *) g_memdup2 (src_user_metadata,
      sizeof (NvDsObjEncOutParams));
  dst_user_metadata->outBuffer =
      (uint8_t *) g_memdup2 (src_user_metadata->outBuffer,
      src_user_metadata->outLen);
  return (gpointer) dst_user_metadata;
}

/* release function set by user. "data" holds a pointer to NvDsUserMeta */
void
nvds_release_image_encode_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  if (user_meta->user_meta_data) {
    NvDsObjEncOutParams *src_user_metadata =
        (NvDsObjEncOutParams *) user_meta->user_meta_data;
    g_free (src_user_metadata->outBuffer);
    src_user_metadata->outBuffer = NULL;
    g_free (user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
  }
}

NvDsBufferList *
nvds_acquire_buffer_from_pool (NvDsObjEncCtxHandle ctx)
{
  NvDsObjEncCtx *pctx = (NvDsObjEncCtx *) ctx;
  g_mutex_lock (&(pctx->empty_list_mutex));
  NvDsBufferList *element_from_list = g_list_first (pctx->empty_list);
  while (element_from_list == NULL) {
    g_cond_wait (&(pctx->buffer_cond), &(pctx->empty_list_mutex));
    element_from_list = g_list_first (pctx->empty_list);
  }
  pctx->empty_list = g_list_remove_link (pctx->empty_list, element_from_list);
  pctx->num_empty_elements--;
  pctx->full_list = g_list_concat (element_from_list, pctx->full_list);
  pctx->num_full_elements++;
  g_mutex_unlock (&(pctx->empty_list_mutex));
  return element_from_list;
}

void
nvds_release_buffer_to_pool (NvDsObjEncCtxHandle ctx,
    NvDsBufferList * element_from_list)
{
  NvDsObjEncCtx *pctx = (NvDsObjEncCtx *) ctx;
  g_mutex_lock (&(pctx->empty_list_mutex));
  pctx->full_list = g_list_remove_link (pctx->full_list, element_from_list);
  pctx->num_full_elements--;
  pctx->empty_list = g_list_concat (pctx->empty_list, element_from_list);
  pctx->num_empty_elements++;
  g_cond_broadcast (&(pctx->buffer_cond));
  g_cond_broadcast (&(pctx->frame_finish_cond));
  g_mutex_unlock (&(pctx->empty_list_mutex));
}

NvDsObjEncCtxHandle nvds_obj_enc_create_context (int gpu_id)
{
  NvDsObjEncCtx *ctx = (NvDsObjEncCtx *) calloc (1, sizeof (NvDsObjEncCtx));

  ctx->gpu_id = gpu_id;

  cudaError_t cudaReturn = cudaSetDevice(ctx->gpu_id);
  if (cudaReturn != cudaSuccess) {
    g_print ("nvll_batch_jpeg_enc: Failed to set gpu-id with error: %s\n", cudaGetErrorName (cudaReturn));
    g_free (ctx);
    return FALSE;
  }

  cudaReturn = cudaStreamCreateWithFlags (&(ctx->cuda_stream), cudaStreamNonBlocking);
  if (cudaReturn != cudaSuccess) {
    g_print ("nvll_batch_jpeg_enc: Failed to create cuda stream for buffer conversion: %s\n",
                cudaGetErrorName (cudaReturn));
    g_free (ctx);
    return FALSE;
  }

  /* Check to detect igpu platforms */
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, ctx->gpu_id);
  ctx->is_integrated = prop.integrated;

  /* Treat iGPU as dGPU if VIC is absent (For Ex. - DIGITs) */
  if (ctx->is_integrated) {
    NvBufSurfaceDeviceInfo dev_info = {0};
    if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
      if (!dev_info.isVicPresent) {
        ctx->is_integrated = 0;
      }
    }
  }

  /* For unified/pinned memory */
  ctx->supports_unified_mem = 1;

  ctx->handle_nvml_lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
  if (ctx->handle_nvml_lib)
  {
    nvmlInit_fn = (typeof(&nvmlInit)) dlsym(ctx->handle_nvml_lib, "nvmlInit");
    nvmlDeviceGetHandleByIndex_fn = (typeof(&nvmlDeviceGetHandleByIndex)) dlsym(ctx->handle_nvml_lib, "nvmlDeviceGetHandleByIndex");
    nvmlDeviceGetBrand_fn = (typeof(&nvmlDeviceGetBrand)) dlsym(ctx->handle_nvml_lib, "nvmlDeviceGetBrand");
    nvmlShutdown_fn = (typeof(&nvmlShutdown)) dlsym(ctx->handle_nvml_lib, "nvmlShutdown");

    nvmlDevice_t device;
    nvmlReturn_t nvmlResult;
    nvmlBrandType_t type;
    nvmlInit_fn();
    nvmlResult = nvmlDeviceGetHandleByIndex_fn(ctx->gpu_id, &device);
    if (nvmlResult != NVML_SUCCESS){
      g_print("nvmlDeviceGetHandleByIndex failed with error(%d) %s\n", nvmlResult, __func__);
      g_free (ctx);
      return FALSE;
    }
    nvmlResult = nvmlDeviceGetBrand_fn(device, &type);
    if (nvmlResult != NVML_SUCCESS){
      g_print("nvmlDeviceGetBrand failed with error(%d) %s\n", nvmlResult, __func__);
      g_free (ctx);
      return FALSE;
    }
    /* Use Pinned memory for vGPU */
    if (type == NVML_BRAND_NVIDIA_VCS || type == NVML_BRAND_GRID){
      ctx->supports_unified_mem = 0;
    }
    nvmlShutdown_fn();
  }

  /* No of cores */
  ctx->num_cores = sysconf (_SC_NPROCESSORS_ONLN);
  g_mutex_init (&(ctx->empty_list_mutex));
  g_cond_init (&(ctx->buffer_cond));
  g_cond_init (&(ctx->frame_finish_cond));

  g_mutex_init (&(ctx->enc_list_mutex));
  g_cond_init (&(ctx->signal_encode));

  /* Create a threadpool */
  g_thread_pool_set_max_unused_threads (-1);
#ifdef __aarch64__
  if(ctx->is_integrated) {
      NvDsOpenJpegLib(ctx->handle_jpeg_lib);
      ctx->thread_pool =
          g_thread_pool_new ((GFunc) jpeg_convert, NULL, ctx->num_cores, FALSE,
          NULL);
      init_encoder_instances (ctx);
  }
  else
#endif
  {
    NvDsOpenJpegLib_DGPU(ctx->handle_jpeg_lib);
    ctx->thread_pool =
        g_thread_pool_new ((GFunc) jpeg_convert_dgpu, NULL, ctx->num_cores, FALSE,
        NULL);
    init_encoder_instances_dgpu (ctx);
  }

  ctx->buffers_alloted = FALSE;
  ctx->num_empty_elements = 0;
  ctx->num_full_elements = 0;
  return ctx;
}

gboolean
allocate_buffers (NvDsObjEncCtxHandle ctx, NvBufSurface * ip_surf, int width,
    int height)
{
  NvDsObjEncCtx *pctx = (NvDsObjEncCtx *) ctx;

  /* Creating (Populating) params for an NvBufSurface */
  NvBufSurfaceCreateParams create_params = { 0 };
  create_params.gpuId = ip_surf->gpuId;
  create_params.width = width;
  create_params.height = height;
  create_params.layout = ip_surf->surfaceList->layout;

  if (ctx->is_integrated) {
    /* On pre-T264 Tegra (Orin and older), VIC does not accept
     * NVBUF_MEM_CUDA_DEVICE; it requires NVBUF_MEM_SURFACE_ARRAY.
     * Thor (T264+) added CUDA-DMABUF interop in VIC. */
    NvBufSurfaceDeviceInfo dev_info = {0};
    NvBufSurfaceGetDeviceInfo(&dev_info);
    if (dev_info.platformType < NVBUF_PLATFORM_TYPE_T264) {
      create_params.memType = NVBUF_MEM_SURFACE_ARRAY;
    } else {
      create_params.memType = NVBUF_MEM_CUDA_DEVICE;
    }
    create_params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
  } else {
    if (ctx->supports_unified_mem) {
      create_params.memType = NVBUF_MEM_CUDA_UNIFIED;
    } else {
      create_params.memType = NVBUF_MEM_CUDA_PINNED;
    }
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGB;
  }
  create_params.size = 0;
  /* Pool of scratchbuffers */
  pctx->scratch_buffer =
      (NvBufSurface **) calloc (pctx->num_cores, sizeof (NvBufSurface *));
  for (int i = 0; i < pctx->num_cores; i++) {
    if (NvBufSurfaceCreate (&((pctx->scratch_buffer)[i]), 1,
            &create_params) != 0) {
      /* g_print ("Error: Could not allocate internal buffer for image dump"); */
      pctx->buffers_alloted = FALSE;
      return FALSE;
    }
    pctx->empty_list =
        g_list_prepend (pctx->empty_list, (gpointer) (pctx->scratch_buffer)[i]);
    pctx->num_empty_elements++;
  }
  pctx->buffers_alloted = TRUE;
  return TRUE;
}

gboolean
deallocate_buffers (NvDsObjEncCtxHandle ctx)
{
  NvDsObjEncCtx *pctx = (NvDsObjEncCtx *) ctx;
  /* Destroy the pool of scratchbuffers */
  for (int i = 0; i < pctx->num_cores; i++) {
    if (NvBufSurfaceDestroy ((pctx->scratch_buffer)[i]) != 0) {
      /*  g_print ("Error: Could not de-allocate internal buffer for image dump"); */
      return FALSE;
    }
  }
  g_list_free (pctx->empty_list);
  pctx->empty_list = NULL;
  pctx->num_empty_elements = 0;
  pctx->buffers_alloted = FALSE;
  return TRUE;
}

bool
nvds_obj_enc_process (NvDsObjEncCtxHandle ctx,
    NvDsObjEncUsrArgs * src_user_data, NvBufSurface * ip_surf,
    NvDsObjectMeta * obj_meta, NvDsFrameMeta * frame_meta)
{
  struct timeval t1;
  gettimeofday(&t1, NULL);

  g_return_val_if_fail (src_user_data != NULL, FALSE);
  NvDsObjEncCtx *pctx = (NvDsObjEncCtx *) ctx;
  NvDsObjEncUsrArgs *user_data = (NvDsObjEncUsrArgs *) g_memdup2 (src_user_data,
      sizeof (NvDsObjEncUsrArgs));
  int frame_surface_list_index = frame_meta->batch_id;
  int frame_width = ip_surf->surfaceList[frame_surface_list_index].width;
  int frame_height = ip_surf->surfaceList[frame_surface_list_index].height;
  NvBufSurfTransform_Error err;

  NvBufSurfTransformConfigParams transform_config_params;
  transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;

  if (user_data == NULL) {
    g_print ("Error: Could not allocate memory for user data\n");
    return FALSE;
  }

  if((ip_surf->memType == NVBUF_MEM_CUDA_DEVICE) && (ip_surf->gpuId != (uint32_t) pctx->gpu_id))
  {
    g_print ("Error (batch_jpeg_enc): Transformation of CUDA_DEVICE memory on "
              "different gpus is not allowed.\n"
              "Please use either UNIFIED or PINNED memory for the transformation or\n"
              "do the transformation on the same gpus.\n\n");
    g_free (user_data);
    return FALSE;
  }

  if (!user_data->isFrame && obj_meta != NULL) {
    if (((int) obj_meta->rect_params.width <= 0)
        || ((int) obj_meta->rect_params.height <= 0)
        || ((int) obj_meta->rect_params.left < 0)
        || ((int) obj_meta->rect_params.top < 0)) {
      g_print
          ("Error: Object dimensions cannot be non-positive. Object not encoded.\n");
      g_free (user_data);
      return FALSE;
    }

    if ((frame_width < (int) obj_meta->rect_params.width)
        || (frame_height < (int) obj_meta->rect_params.height)) {
      g_print
          ("Error: Object dimensions are greater than frame dimensions. Object not encoded.\n");
      g_free (user_data);
      return FALSE;
    } else if ((frame_width <
            (int) obj_meta->rect_params.width + (int) obj_meta->rect_params.left)
        || (frame_height <
            (int) obj_meta->rect_params.height +
            (int) obj_meta->rect_params.top)) {
      g_print
          ("Error: Object dimensions are out of frame boundary. Object not encoded.\n");
      g_free (user_data);
      return FALSE;
    }

    if(ctx->is_integrated && (transform_config_params.compute_mode == NvBufSurfTransformCompute_VIC 
      || transform_config_params.compute_mode == NvBufSurfTransformCompute_Default)) {
      if ((int) obj_meta->rect_params.width < 16 || (int) obj_meta->rect_params.height < 16) {
        g_print("Warning: Object dimensions less than 16 are not supported by VIC HW. Object not encoded. Use GPU scaling instead\n");
        g_free (user_data);
        return FALSE;
      }
    }
  }

  if (user_data->scaleImg) {
    if ((int) user_data->scaledHeight <= 0 || (int) user_data->scaledWidth <= 0) {
      g_print
          ("Error: Scaled object dimensions cannot be non-positive. Object not encoded.\n");
      g_free (user_data);
      return FALSE;
    }

    if ((frame_height < (int) user_data->scaledHeight)
        || (frame_width < (int) user_data->scaledWidth)) {
      g_print
          ("Error: Scaled object dimensions are greater than frame dimensions. Object not encoded.\n");
      g_free (user_data);
      return FALSE;
    }

#ifndef __x86_64__
    if (ctx->is_integrated && !user_data->isFrame) {
        if (((int) obj_meta->rect_params.width * 16 < (int) user_data->scaledWidth)
         || ((int) obj_meta->rect_params.height * 16 < (int) user_data->scaledHeight)) {
            g_print
                ("Error: Scaled object dimensions can't be more than 16 * object dimensions. Object not encoded.\n");
            g_free (user_data);
            return FALSE;
        }
    }
#endif
  }

  /* Allocate Buffers */
  uint max_frame_width = (get_max (1280, frame_width) + 1) & ~1;
  uint max_frame_height = (get_max (720, frame_height) + 1) & ~1;

  if (pctx->buffers_alloted == FALSE) {
    if (!allocate_buffers (ctx, ip_surf, max_frame_width, max_frame_height)) {
      /*  g_print ("scratch buffer allocation failed"); */
      g_free (user_data);
      return FALSE;
    }
  } else if (max_frame_width > (pctx->scratch_buffer)[0]->surfaceList->width
      || max_frame_height > (pctx->scratch_buffer)[0]->surfaceList->height) {
    /* Twice the previous value */
    max_frame_width =
        (get_max (max_frame_width,
        2 * (pctx->scratch_buffer)[0]->surfaceList->width) + 1) & ~1;
    max_frame_height =
        (get_max (max_frame_height,
        2 * (pctx->scratch_buffer)[0]->surfaceList->height) + 1) & ~1;

    /* Make sure that all the existing crops have been encoded and all the
     * scratch buffers have been returned back to the queue.
     */
    nvds_obj_enc_finish (ctx);
    if (!deallocate_buffers (ctx)) {
      /* g_print ("scratch buffer de-allocation failed"); */
      g_free (user_data);
      return FALSE;
    }
    if (!allocate_buffers (ctx, ip_surf, max_frame_width, max_frame_height)) {
      /*  g_print ("scratch buffer allocation failed"); */
      g_free (user_data);
      return FALSE;
    }
  }

#ifndef __x86_64__
  if(!pctx->is_integrated)
    transform_config_params.compute_mode = NvBufSurfTransformCompute_GPU;
#endif
  transform_config_params.gpu_id = pctx->gpu_id;
  transform_config_params.cuda_stream = pctx->cuda_stream;
  err = NvBufSurfTransformSetSessionParams (&transform_config_params);

  if (err != NvBufSurfTransformError_Success) {
    g_print ("nvll_batch_jpeg_enc: NvBufSurfTransformSetSessionParams failed with error %d", err);
    g_free (user_data);
    return FALSE;
  }

  /* Acquire buffer from pool */
  NvDsBufferList *temp_list = nvds_acquire_buffer_from_pool (ctx);
  NvBufSurface *allocated_buffer = (NvBufSurface *) temp_list->data;
  /* Memset the memory */
  NvBufSurfaceMemSet (allocated_buffer, 0, 0, 0);

  /* Copy a cropped image data to scratchbuffer */
  NvBufSurfTransformParams transform_params;
  NvBufSurfTransformRect src_rect;
  NvBufSurfTransformRect dst_rect;

  if(user_data->isFrame) {
    src_rect.left = 0;
    src_rect.top = 0;
    src_rect.width = frame_width;
    src_rect.height = frame_height;

    dst_rect.left = 0;
    dst_rect.top = 0;
    if (!user_data->scaleImg) {
      dst_rect.width = frame_width;
      dst_rect.height = frame_height;
    } else {
      dst_rect.width = (int) user_data->scaledWidth;
      dst_rect.height = (int) user_data->scaledHeight;
    }
  }
  else {
    src_rect.left = (int) obj_meta->rect_params.left;
    src_rect.top = (int) obj_meta->rect_params.top;
    src_rect.width = (int) obj_meta->rect_params.width;
    src_rect.height = (int) obj_meta->rect_params.height;

    dst_rect.left = 0;
    dst_rect.top = 0;
    if (!user_data->scaleImg) {
      dst_rect.width = (int) obj_meta->rect_params.width;
      dst_rect.height = (int) obj_meta->rect_params.height;
    } else {
      dst_rect.width = (int) user_data->scaledWidth;
      dst_rect.height = (int) user_data->scaledHeight;
    }
  }
  dst_rect.width = (dst_rect.width + 1) & ~1;
  dst_rect.height = (dst_rect.height + 1) & ~1;

  /* Set the transform parameters */
  transform_params.src_rect = &src_rect;
  transform_params.dst_rect = &dst_rect;
  transform_params.transform_flag =
      NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
      NVBUFSURF_TRANSFORM_CROP_DST;
  transform_params.transform_filter = NvBufSurfTransformInter_Default;

  /* Need to pass pointer to the video stream containing the object */
  NvBufSurface frame_surface_list = *ip_surf;
  (&frame_surface_list)->batchSize = 1;
  (&frame_surface_list)->numFilled = 1;
  (&frame_surface_list)->surfaceList =
      ip_surf->surfaceList + frame_surface_list_index;

  NvBufSurfTransform (&frame_surface_list, allocated_buffer, &transform_params);

  int res;

  /* Assign image name */
  if (!strcmp (user_data->fileNameImg, "")) {
    /* Frame */
    if(user_data->isFrame) {
      res = snprintf (user_data->fileNameImg, FILE_NAME_SIZE,
          "frame_%d_%d_%dx%d.jpg",
          frame_meta->frame_num, frame_surface_list_index,
          dst_rect.width, dst_rect.height);
    } else { /* Object */
      res = snprintf (user_data->fileNameImg, FILE_NAME_SIZE,
          "%d_%d_%d_%s_%dx%d.jpg",
          frame_meta->frame_num, frame_surface_list_index,
          user_data->objNum, obj_meta->obj_label, dst_rect.width, dst_rect.height);
    }
    if (res >= FILE_NAME_SIZE)
      fprintf (stderr,
          "Image file name too long (%d). Should be lower than %d.\n", res,
          FILE_NAME_SIZE);
  }

  /* Function arguments for Jpeg Convert */
  NvDsJpegConvArgs *funcArgs =
      (NvDsJpegConvArgs *) calloc (1, sizeof (NvDsJpegConvArgs));
  funcArgs->scratchBuffer = allocated_buffer;
  funcArgs->ctx = ctx;
  funcArgs->tempList = temp_list;
  funcArgs->frameMeta = frame_meta;
  funcArgs->objMeta = obj_meta;
  funcArgs->userData = user_data;
  funcArgs->width = (uint) dst_rect.width;
  funcArgs->height = (uint) dst_rect.height;
  if (user_data->calcEncodeTime) {
    funcArgs->initTimeSec = t1.tv_sec;
    funcArgs->initTimeUsec = t1.tv_usec;
  }
  /* Add job to task queue */
  g_thread_pool_push (pctx->thread_pool, funcArgs, NULL);
  return TRUE;
}

void
nvds_obj_enc_finish (NvDsObjEncCtxHandle ctx)
{
  NvDsObjEncCtx *pctx = (NvDsObjEncCtx *) ctx;
  if (pctx->buffers_alloted) {
    g_mutex_lock (&(pctx->empty_list_mutex));
    while (pctx->num_empty_elements != pctx->num_cores) {
      g_cond_wait (&(pctx->frame_finish_cond), &(pctx->empty_list_mutex));
    }
    g_mutex_unlock (&(pctx->empty_list_mutex));
  }
}

void nvds_obj_enc_destroy_context (NvDsObjEncCtxHandle ctx)
{
  NvDsObjEncCtx *pctx = (NvDsObjEncCtx *) ctx;
  if (pctx->total_encodes > 0) {
    pctx->avg_encode_time = pctx->total_encode_time/pctx->total_encodes;
    printf("Avg encode time - %Lf ms\n", pctx->avg_encode_time);
  }
  if (pctx->handle_nvml_lib) {
    dlclose(pctx->handle_nvml_lib);
    pctx->handle_nvml_lib = NULL;
  }
  if (pctx->buffers_alloted)
    deallocate_buffers (ctx);

#ifdef __aarch64__
  if (pctx->is_integrated) {
    destroy_encoder_instances (ctx);
    NvDsCloseJpegLib(pctx->handle_jpeg_lib);
  }
  else
#endif
  {
    destroy_encoder_instances_dgpu (ctx);
    NvDsCloseJpegLib_DGPU(pctx->handle_jpeg_lib);
  }

  pctx->handle_jpeg_lib = NULL;

  g_thread_pool_stop_unused_threads ();
  g_thread_pool_free (pctx->thread_pool, FALSE, TRUE);
  g_mutex_clear (&(pctx->empty_list_mutex));
  g_mutex_clear (&(pctx->enc_list_mutex));
  g_cond_clear (&(pctx->buffer_cond));
  g_cond_clear (&(pctx->frame_finish_cond));
  g_cond_clear (&(pctx->signal_encode));

  cudaStreamDestroy (pctx->cuda_stream);
  pctx->cuda_stream = NULL;
  free (ctx);
  ctx = NULL;
}
