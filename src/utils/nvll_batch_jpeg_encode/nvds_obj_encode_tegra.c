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
 *
 * version: 0.1
 */


#include <glib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <dlfcn.h>
#include "nvdsmeta.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "jpeglib.h"
#include "nvds_obj_encode.h"
#include "nvds_obj_encode_common.h"
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef JPEG_LIB_VERSION
#undef JPEG_LIB_VERSION
#endif

// Enable backward compatibility
#define JPEG_LIB_VERSION  80

static typeof(&jpeg_std_error) jpeg_std_error_fn;
static typeof(&jpeg_CreateCompress) jpeg_CreateCompress_fn;
static typeof(&jpeg_suppress_tables) jpeg_suppress_tables_fn;
static typeof(&jpeg_mem_dest) jpeg_mem_dest_fn;
static typeof(&jpeg_set_defaults) jpeg_set_defaults_fn;
static typeof(&jpeg_set_hardware_acceleration_parameters_enc)
    jpeg_set_hardware_acceleration_parameters_enc_fn;
static typeof(&jpeg_start_compress) jpeg_start_compress_fn;
static typeof(&jpeg_write_raw_data) jpeg_write_raw_data_fn;
static typeof(&jpeg_finish_compress) jpeg_finish_compress_fn;
static typeof(&jpeg_destroy_compress) jpeg_destroy_compress_fn;
static typeof(&jpeg_set_quality) jpeg_set_quality_fn;

void NvDsCloseJpegLib (void * handle_lib)
{
  if (handle_lib) {
    dlclose(handle_lib);
    handle_lib = NULL;
  }
}

void NvDsOpenJpegLib (void * handle_lib)
{
  handle_lib = dlopen("libnvmm_jpeg.so", RTLD_LAZY);
  if (!handle_lib)
  {
    g_print("Failed to open lib libnvmm_jpeg.so \n");
    return;
  }

  jpeg_std_error_fn = (typeof(&jpeg_std_error)) dlsym(handle_lib, "jpeg_std_error");
  jpeg_CreateCompress_fn = (typeof(&jpeg_CreateCompress)) dlsym(handle_lib, "jpeg_CreateCompress");
  jpeg_suppress_tables_fn = (typeof(&jpeg_suppress_tables)) dlsym(handle_lib, "jpeg_suppress_tables");
  jpeg_mem_dest_fn = (typeof(&jpeg_mem_dest)) dlsym(handle_lib, "jpeg_mem_dest");
  jpeg_set_defaults_fn = (typeof(&jpeg_set_defaults)) dlsym(handle_lib, "jpeg_set_defaults");
  jpeg_set_hardware_acceleration_parameters_enc_fn =
    (typeof(&jpeg_set_hardware_acceleration_parameters_enc))
    dlsym(handle_lib, "jpeg_set_hardware_acceleration_parameters_enc");
  jpeg_start_compress_fn = (typeof(&jpeg_start_compress)) dlsym(handle_lib, "jpeg_start_compress");
  jpeg_write_raw_data_fn = (typeof(&jpeg_write_raw_data)) dlsym(handle_lib, "jpeg_write_raw_data");
  jpeg_finish_compress_fn = (typeof(&jpeg_finish_compress)) dlsym(handle_lib, "jpeg_finish_compress");
  jpeg_destroy_compress_fn = (typeof(&jpeg_destroy_compress)) dlsym(handle_lib, "jpeg_destroy_compress");
  jpeg_set_quality_fn = (typeof(&jpeg_set_quality)) dlsym(handle_lib, "jpeg_set_quality");
}

void init_encoder_instances (NvDsObjEncCtx *ctx) {
  for (int i = 0; i < ctx->num_cores; i++) {
    struct jpeg_error_mgr *jerr = NULL;
    struct jpeg_compress_struct *cinfo = NULL;
    jerr = (struct jpeg_error_mgr *) malloc(sizeof(struct jpeg_error_mgr));
    cinfo = (struct jpeg_compress_struct *) malloc(sizeof(struct jpeg_compress_struct));

    cinfo->err = jpeg_std_error_fn (jerr);
    jpeg_CreateCompress_fn (cinfo, JPEG_LIB_VERSION, (size_t) sizeof(struct jpeg_compress_struct));
    jpeg_suppress_tables_fn (cinfo, TRUE);

    ctx->free_list =
        g_list_prepend (ctx->free_list, (gpointer) cinfo);
  }
}

void destroy_encoder_instances (NvDsObjEncCtxHandle ctx) {
  NvDsObjEncCtx *pctx = (NvDsObjEncCtx *) ctx;
  g_mutex_lock (&(pctx->enc_list_mutex));

  GList *element_from_list = g_list_first (pctx->free_init_list);
  while (element_from_list) {
    pctx->free_init_list = g_list_remove_link (pctx->free_init_list, element_from_list);
    struct jpeg_compress_struct *cinfo = element_from_list->data;
    struct jpeg_error_mgr *jerr = cinfo->err;
    g_free (jerr);
    jpeg_destroy_compress_fn (cinfo);
    g_free (cinfo);
    element_from_list = g_list_first (pctx->free_init_list);
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
acquire_encoder_instance (NvDsObjEncCtxHandle ctx) {

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
release_encoder_instance (NvDsObjEncCtxHandle ctx,
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
jpeg_convert (NvDsJpegConvArgs * conv_args)
{
  uint8_t *out_buffer = NULL;
  out_buffer = (uint8_t *) malloc (4 * 1024 * 1024);
  uint64_t out_len = 4 * 1024 * 1024;

  GList * temp_list = acquire_encoder_instance (conv_args->ctx);
  struct jpeg_compress_struct *cinfo = (struct jpeg_compress_struct *) temp_list->data;
  jpeg_mem_dest_fn (cinfo, &out_buffer, &out_len);

  cinfo->fd = conv_args->scratchBuffer->surfaceList[0].bufferDesc;
  cinfo->IsVendorbuf = TRUE;
  cinfo->raw_data_in = TRUE;
  cinfo->in_color_space = JCS_YCbCr;

  cinfo->image_width = (int) conv_args->width;
  cinfo->image_height = (int) conv_args->height;

  jpeg_set_defaults_fn (cinfo);
  if (conv_args->userData->quality > 0 &&
      conv_args->userData->quality <= 100)
    jpeg_set_quality_fn(cinfo, conv_args->userData->quality, TRUE);
  jpeg_set_hardware_acceleration_parameters_enc_fn (cinfo, TRUE, out_len, 0, 0);
  jpeg_start_compress_fn (cinfo, TRUE);

  jpeg_write_raw_data_fn (cinfo, NULL, 0);

  jpeg_finish_compress_fn (cinfo);

  release_encoder_instance (conv_args->ctx, temp_list);

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
