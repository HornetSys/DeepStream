/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * Dynamic loading implementation for libgstnvdsseimeta and libgstnvcustomhelper
 */

#include <gst/gst.h>
#include <dlfcn.h>
#include <stdio.h>

#include "gstnvdsseimeta.h"
#include "gst-nvcustomevent.h"
#include "nvhelper_dyn.h"

/* Library handles */
static void *sei_meta_lib = NULL;
static void *custom_event_lib = NULL;
static gboolean init_done = FALSE;

/* Function pointer types for SEI meta library */
typedef GType (*fp_gst_video_sei_meta_api_get_type)(void);
typedef const GstMetaInfo* (*fp_gst_video_sei_meta_get_info)(void);

/* Function pointer types for custom event library */
typedef void (*fp_gst_nvevent_parse_dec_drop_frame_interval_update)(GstEvent*, gchar**, guint*);
typedef void (*fp_gst_nvevent_parse_dec_skip_frame_update)(GstEvent*, gchar**, guint*);
typedef void (*fp_gst_nvevent_parse_dec_enable_low_latency_mode)(GstEvent*, gchar**, gint*);
typedef void (*fp_gst_nvevent_parse_enc_bitrate_update)(GstEvent*, gchar**, guint*);
typedef void (*fp_gst_nvevent_parse_enc_force_idr)(GstEvent*, gchar**, gint*);
typedef void (*fp_gst_nvevent_parse_enc_force_intra)(GstEvent*, gchar**, gint*);
typedef void (*fp_gst_nvevent_parse_enc_iframeinterval_update)(GstEvent*, gchar**, guint*);

/* Function pointers for SEI meta */
static fp_gst_video_sei_meta_api_get_type real_sei_meta_api_get_type = NULL;
static fp_gst_video_sei_meta_get_info real_sei_meta_get_info = NULL;

/* Function pointers for custom events */
static fp_gst_nvevent_parse_dec_drop_frame_interval_update real_parse_dec_drop_frame = NULL;
static fp_gst_nvevent_parse_dec_skip_frame_update real_parse_dec_skip_frame = NULL;
static fp_gst_nvevent_parse_dec_enable_low_latency_mode real_parse_dec_low_latency = NULL;
static fp_gst_nvevent_parse_enc_bitrate_update real_parse_enc_bitrate = NULL;
static fp_gst_nvevent_parse_enc_force_idr real_parse_enc_force_idr = NULL;
static fp_gst_nvevent_parse_enc_force_intra real_parse_enc_force_intra = NULL;
static fp_gst_nvevent_parse_enc_iframeinterval_update real_parse_enc_iframeinterval = NULL;

void
nvhelper_dyn_init(void)
{
  if (init_done)
    return;

  init_done = TRUE;

  sei_meta_lib = dlopen("libgstnvdsseimeta.so", RTLD_NOW | RTLD_GLOBAL);
  if (!sei_meta_lib) {
    sei_meta_lib = dlopen("libgstnvdsseimeta.so.1.0.0", RTLD_NOW | RTLD_GLOBAL);
  }

  if (sei_meta_lib) {
    real_sei_meta_api_get_type = (fp_gst_video_sei_meta_api_get_type)
        dlsym(sei_meta_lib, "gst_video_sei_meta_api_get_type");
    real_sei_meta_get_info = (fp_gst_video_sei_meta_get_info)
        dlsym(sei_meta_lib, "gst_video_sei_meta_get_info");

    if (!real_sei_meta_api_get_type || !real_sei_meta_get_info) {
      GST_WARNING("Failed to resolve symbols from libgstnvdsseimeta.so");
      real_sei_meta_api_get_type = NULL;
      real_sei_meta_get_info = NULL;
    } else {
      GST_DEBUG("Successfully loaded libgstnvdsseimeta.so");
    }
  } else {
    GST_DEBUG("libgstnvdsseimeta.so not available: %s", dlerror());
  }

  custom_event_lib = dlopen("libgstnvcustomhelper.so", RTLD_NOW | RTLD_GLOBAL);
  if (custom_event_lib) {
    real_parse_dec_drop_frame = (fp_gst_nvevent_parse_dec_drop_frame_interval_update)
        dlsym(custom_event_lib, "gst_nvevent_parse_dec_drop_frame_interval_update");
    real_parse_dec_skip_frame = (fp_gst_nvevent_parse_dec_skip_frame_update)
        dlsym(custom_event_lib, "gst_nvevent_parse_dec_skip_frame_update");
    real_parse_dec_low_latency = (fp_gst_nvevent_parse_dec_enable_low_latency_mode)
        dlsym(custom_event_lib, "gst_nvevent_parse_dec_enable_low_latency_mode");
    real_parse_enc_bitrate = (fp_gst_nvevent_parse_enc_bitrate_update)
        dlsym(custom_event_lib, "gst_nvevent_parse_enc_bitrate_update");
    real_parse_enc_force_idr = (fp_gst_nvevent_parse_enc_force_idr)
        dlsym(custom_event_lib, "gst_nvevent_parse_enc_force_idr");
    real_parse_enc_force_intra = (fp_gst_nvevent_parse_enc_force_intra)
        dlsym(custom_event_lib, "gst_nvevent_parse_enc_force_intra");
    real_parse_enc_iframeinterval = (fp_gst_nvevent_parse_enc_iframeinterval_update)
        dlsym(custom_event_lib, "gst_nvevent_parse_enc_iframeinterval_update");

    GST_DEBUG("Successfully loaded libgstnvcustomhelper.so");
  } else {
    GST_DEBUG("libgstnvcustomhelper.so not available: %s", dlerror());
  }
}

void
nvhelper_dyn_deinit(void)
{
  if (!init_done)
    return;

  real_sei_meta_api_get_type = NULL;
  real_sei_meta_get_info = NULL;
  real_parse_dec_drop_frame = NULL;
  real_parse_dec_skip_frame = NULL;
  real_parse_dec_low_latency = NULL;
  real_parse_enc_bitrate = NULL;
  real_parse_enc_force_idr = NULL;
  real_parse_enc_force_intra = NULL;
  real_parse_enc_iframeinterval = NULL;

  if (sei_meta_lib) {
    dlclose(sei_meta_lib);
    sei_meta_lib = NULL;
  }

  if (custom_event_lib) {
    dlclose(custom_event_lib);
    custom_event_lib = NULL;
  }

  init_done = FALSE;
}

__attribute__((constructor))
static void
nvhelper_dyn_auto_init(void)
{
  nvhelper_dyn_init();
}

__attribute__((destructor))
static void
nvhelper_dyn_fini(void)
{
  nvhelper_dyn_deinit();
}

/* SEI Meta wrapper functions */
GType
gst_video_sei_meta_api_get_type(void)
{
  if (real_sei_meta_api_get_type)
    return real_sei_meta_api_get_type();
  return G_TYPE_INVALID;
}

const GstMetaInfo *
gst_video_sei_meta_get_info(void)
{
  if (real_sei_meta_get_info)
    return real_sei_meta_get_info();
  return NULL;
}

/* Custom event wrapper functions */
void
gst_nvevent_parse_dec_drop_frame_interval_update(GstEvent *event, gchar **stream_id, guint *interval)
{
  if (real_parse_dec_drop_frame) {
    real_parse_dec_drop_frame(event, stream_id, interval);
  } else {
    if (stream_id) *stream_id = NULL;
    if (interval) *interval = 0;
  }
}

void
gst_nvevent_parse_dec_skip_frame_update(GstEvent *event, gchar **stream_id, guint *frame_type)
{
  if (real_parse_dec_skip_frame) {
    real_parse_dec_skip_frame(event, stream_id, frame_type);
  } else {
    if (stream_id) *stream_id = NULL;
    if (frame_type) *frame_type = 0;
  }
}

void
gst_nvevent_parse_dec_enable_low_latency_mode(GstEvent *event, gchar **stream_id, gint *enable)
{
  if (real_parse_dec_low_latency) {
    real_parse_dec_low_latency(event, stream_id, enable);
  } else {
    if (stream_id) *stream_id = NULL;
    if (enable) *enable = 0;
  }
}

void
gst_nvevent_parse_enc_bitrate_update(GstEvent *event, gchar **stream_id, guint *bitrate)
{
  if (real_parse_enc_bitrate) {
    real_parse_enc_bitrate(event, stream_id, bitrate);
  } else {
    if (stream_id) *stream_id = NULL;
    if (bitrate) *bitrate = 0;
  }
}

void
gst_nvevent_parse_enc_force_idr(GstEvent *event, gchar **stream_id, gint *force)
{
  if (real_parse_enc_force_idr) {
    real_parse_enc_force_idr(event, stream_id, force);
  } else {
    if (stream_id) *stream_id = NULL;
    if (force) *force = 0;
  }
}

void
gst_nvevent_parse_enc_force_intra(GstEvent *event, gchar **stream_id, gint *force)
{
  if (real_parse_enc_force_intra) {
    real_parse_enc_force_intra(event, stream_id, force);
  } else {
    if (stream_id) *stream_id = NULL;
    if (force) *force = 0;
  }
}

void
gst_nvevent_parse_enc_iframeinterval_update(GstEvent *event, gchar **stream_id, guint *interval)
{
  if (real_parse_enc_iframeinterval) {
    real_parse_enc_iframeinterval(event, stream_id, interval);
  } else {
    if (stream_id) *stream_id = NULL;
    if (interval) *interval = 0;
  }
}
