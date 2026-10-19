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
 *
 * version: 0.1
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
#ifdef __x86_64__
#include "jpeglib.h"
#else
#include "oss/jpeglib.h"
#endif
#include "nvds_obj_encode.h"
#include "nvds_obj_encode_common.h"

#ifndef JPEG_LIB_VERSION
#define JPEG_LIB_VERSION  90
#endif

static typeof(&jpeg_std_error) jpeg_std_error_fn;
static typeof(&jpeg_CreateCompress) jpeg_CreateCompress_fn;
static typeof(&jpeg_mem_dest) jpeg_mem_dest_fn;
static typeof(&jpeg_set_defaults) jpeg_set_defaults_fn;
static typeof(&jpeg_write_scanlines) jpeg_write_scanlines_fn;
static typeof(&jpeg_start_compress) jpeg_start_compress_fn;
static typeof(&jpeg_finish_compress) jpeg_finish_compress_fn;
static typeof(&jpeg_destroy_compress) jpeg_destroy_compress_fn;
static typeof(&jpeg_set_quality) jpeg_set_quality_fn;


void NvDsCloseJpegLib_DGPU (void * handle_lib)
{
  if (handle_lib) {
    dlclose(handle_lib);
    handle_lib = NULL;
  }
}

void NvDsOpenJpegLib_DGPU (void * handle_lib)
{
  handle_lib = dlopen("libjpeg.so", RTLD_LAZY);
  if(!handle_lib)
  {
    handle_lib = dlopen("libjpeg.so.9.4.0", RTLD_LAZY);
  }
  if (!handle_lib)
  {
    g_print("Failed to open lib libjpeg.so \n");
    return;
  }

  jpeg_std_error_fn = (typeof(&jpeg_std_error)) dlsym(handle_lib, "jpeg_std_error");
  jpeg_CreateCompress_fn = (typeof(&jpeg_CreateCompress)) dlsym(handle_lib, "jpeg_CreateCompress");
  jpeg_mem_dest_fn = (typeof(&jpeg_mem_dest)) dlsym(handle_lib, "jpeg_mem_dest");
  jpeg_set_defaults_fn = (typeof(&jpeg_set_defaults)) dlsym(handle_lib, "jpeg_set_defaults");
  jpeg_write_scanlines_fn = (typeof(&jpeg_write_scanlines)) dlsym(handle_lib, "jpeg_write_scanlines");
  jpeg_start_compress_fn = (typeof(&jpeg_start_compress)) dlsym(handle_lib, "jpeg_start_compress");
  jpeg_finish_compress_fn = (typeof(&jpeg_finish_compress)) dlsym(handle_lib, "jpeg_finish_compress");
  jpeg_destroy_compress_fn = (typeof(&jpeg_destroy_compress)) dlsym(handle_lib, "jpeg_destroy_compress");
  jpeg_set_quality_fn = (typeof(&jpeg_set_quality)) dlsym(handle_lib, "jpeg_set_quality");
}


void init_encoder_instances_dgpu (NvDsObjEncCtx *ctx) {
  for (int i = 0; i < ctx->num_cores; i++) {
    struct jpeg_error_mgr *jerr = NULL;
    struct jpeg_compress_struct *cinfo = NULL;
    jerr = (struct jpeg_error_mgr *) malloc(sizeof(struct jpeg_error_mgr));
    cinfo = (struct jpeg_compress_struct *) malloc(sizeof(struct jpeg_compress_struct));

    cinfo->err = jpeg_std_error_fn (jerr);
    jpeg_CreateCompress_fn (cinfo, JPEG_LIB_VERSION, (size_t) sizeof(struct jpeg_compress_struct));

    ctx->free_list =
        g_list_prepend (ctx->free_list, (gpointer) cinfo);
  }
}

void destroy_encoder_instances_dgpu (NvDsObjEncCtxHandle ctx) {
  NvDsObjEncCtx *pctx = (NvDsObjEncCtx *) ctx;
  g_mutex_lock (&(pctx->enc_list_mutex));

  GList *element_from_list = g_list_first (pctx->free_init_list);
  int num_used = 0;
  while (element_from_list) {
    pctx->free_init_list = g_list_remove_link (pctx->free_init_list, element_from_list);
    struct jpeg_compress_struct *cinfo = element_from_list->data;
    struct jpeg_error_mgr *jerr = cinfo->err;
    g_free (jerr);
    jpeg_destroy_compress_fn (cinfo);
    g_free (cinfo);
    element_from_list = g_list_first (pctx->free_init_list);
    num_used++;
  }
  g_list_free (pctx->free_init_list);
  pctx->free_init_list = NULL;

  element_from_list = g_list_first (pctx->free_list);
  while (element_from_list) {
    pctx->free_list = g_list_remove_link (pctx->free_list, element_from_list);
    struct jpeg_compress_struct *cinfo = element_from_list->data;
    struct jpeg_error_mgr *jerr = cinfo->err;
    g_free (jerr);
    jpeg_destroy_compress_fn (cinfo);
    g_free (cinfo);
    element_from_list = g_list_first (pctx->free_list);
  }
  g_list_free (pctx->free_list);
  pctx->free_list = NULL;

  g_list_free (pctx->occupied_list);
  pctx->occupied_list = NULL;

  g_mutex_unlock (&(pctx->enc_list_mutex));
}

GList *
acquire_encoder_instance_dgpu (NvDsObjEncCtxHandle ctx) {
  NvDsObjEncCtx *pctx = (NvDsObjEncCtx *) ctx;
  g_mutex_lock (&(pctx->enc_list_mutex));
  NvDsBufferList *element_from_list = g_list_first (pctx->free_init_list);
  if (element_from_list == NULL) {
    element_from_list = g_list_first (pctx->free_list);
    if (element_from_list != NULL) {
      pctx->free_list = g_list_remove_link (pctx->free_list, element_from_list);
      pctx->occupied_list = g_list_concat (element_from_list, pctx->occupied_list);
      g_mutex_unlock (&(pctx->enc_list_mutex));
      return element_from_list;
    }
  }
  while (element_from_list == NULL) {
    g_cond_wait (&(pctx->signal_encode), &(pctx->enc_list_mutex));
    element_from_list = g_list_first (pctx->free_init_list);
  }
  pctx->free_init_list = g_list_remove_link (pctx->free_init_list, element_from_list);
  pctx->occupied_list = g_list_concat (element_from_list, pctx->occupied_list);
  g_mutex_unlock (&(pctx->enc_list_mutex));
  return element_from_list;
}

void
release_encoder_instance_dgpu (NvDsObjEncCtxHandle ctx,
    GList * element_from_list)
{
  NvDsObjEncCtx *pctx = (NvDsObjEncCtx *) ctx;
  g_mutex_lock (&(pctx->enc_list_mutex));
  pctx->occupied_list = g_list_remove_link (pctx->occupied_list, element_from_list);
  pctx->free_init_list = g_list_concat (pctx->free_init_list, element_from_list);
  g_cond_broadcast (&(pctx->signal_encode));
  g_mutex_unlock (&(pctx->enc_list_mutex));
}

/* Jpeg Encode from scratchbuffer */

gboolean
jpeg_convert_dgpu (NvDsJpegConvArgs * conv_args)
{
  uint8_t *out_buffer = NULL;
  uint64_t out_len = 0;
  GList * temp_list = acquire_encoder_instance_dgpu (conv_args->ctx);
  struct jpeg_compress_struct *cinfo = (struct jpeg_compress_struct *) temp_list->data;
  jpeg_mem_dest_fn (cinfo, &out_buffer, &out_len);

  int row_stride;
  JSAMPROW row_pointer[1];

  cinfo->image_width = (uint) conv_args->width;   /* image width and height, in pixels */
  cinfo->image_height = (uint) conv_args->height;

  cinfo->input_components = 3;   /* # of color components per pixel */
  cinfo->in_color_space = JCS_RGB;

  jpeg_set_defaults_fn (cinfo);
  if (conv_args->userData->quality > 0 &&
      conv_args->userData->quality <= 100)
    jpeg_set_quality_fn(cinfo, conv_args->userData->quality, TRUE);
  jpeg_start_compress_fn (cinfo, TRUE);

  row_stride = conv_args->scratchBuffer->surfaceList->pitch;
  uint8_t *buffer_ptr = (uint8_t *) conv_args->scratchBuffer->surfaceList->dataPtr;

  while (cinfo->next_scanline < cinfo->image_height) {
    /* jpeg_write_scanlines expects an array of pointers to scanlines.
     * Here the array is only one element long, but you could pass
     * more than one scanline at a time if that's more convenient.
     */
    row_pointer[0] = (buffer_ptr + cinfo->next_scanline * row_stride);
    (void) jpeg_write_scanlines_fn (cinfo, row_pointer, 1);
  }

  jpeg_finish_compress_fn (cinfo);

  release_encoder_instance_dgpu (conv_args->ctx, temp_list);

  /* Write output to file */
  if (conv_args->userData->saveImg == TRUE) {
    jpeg_write_to_file (out_buffer, out_len, conv_args->userData->fileNameImg);
  }

  if (conv_args->userData->attachUsrMeta) {
    /* Attach User Metadata */
    NvDsObjEncOutParams *enc_jpeg_image =
        (NvDsObjEncOutParams *) calloc (1, sizeof (NvDsObjEncOutParams));
    if (!enc_jpeg_image) {
      g_print ("Output buffer allocation failed. Cannot attach Metadata\n");
      free (out_buffer);
    } else {
      enc_jpeg_image->outBuffer = out_buffer;
      enc_jpeg_image->outLen = out_len;

      NvDsUserMeta *user_meta = NULL;
      //NvDsBatchMeta *batch_meta = conv_args->objMeta->base_meta.batch_meta;
      /* Acquire NvDsUserMeta user meta from pool */
      if(conv_args->userData->isFrame) {
        user_meta =
          nvds_acquire_user_meta_from_pool (conv_args->frameMeta->base_meta.batch_meta);
      }
      else {
        user_meta =
          nvds_acquire_user_meta_from_pool (conv_args->objMeta->base_meta.batch_meta);
      }

      NvDsMetaType user_meta_type = NVDS_CROP_IMAGE_META;

      /* Set NvDsUserMeta below */
      user_meta->user_meta_data = (void *) (enc_jpeg_image);
      user_meta->base_meta.meta_type = user_meta_type;
      user_meta->base_meta.copy_func =
          (NvDsMetaCopyFunc) nvds_copy_image_encode_meta;
      user_meta->base_meta.release_func =
          (NvDsMetaReleaseFunc) nvds_release_image_encode_meta;

      if(conv_args->userData->isFrame) {
        /* We want to add NvDsUserMeta to frame level */
        nvds_add_user_meta_to_frame (conv_args->frameMeta, user_meta);
      }
      else {
        /* We want to add NvDsUserMeta to obj level */
        nvds_add_user_meta_to_obj (conv_args->objMeta, user_meta);
      }
    }
  } else {
    free (out_buffer);
  }
  nvds_release_buffer_to_pool (conv_args->ctx, conv_args->tempList);
  if (conv_args->userData->calcEncodeTime) {
    calculate_encode_time(conv_args);
  }
  free (conv_args->userData);
  conv_args->userData = NULL;
  free (conv_args);
  conv_args = NULL;
  return TRUE;
}
