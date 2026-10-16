/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gst-nvquery.h"
#include "gst-nvquery-internal.h"

#define QUERY_NUM_SURFACES_PER_BUFFER "num-surfaces-per-buffers"
#define QUERY_NTP_SYNC "nvds-ntp-sync"
#define QUERY_TEXT_EMBEDDING "embedding"
#define QUERY_IMAGE_EMBEDDING "image-embedding"

GstQuery *
gst_nvquery_batch_size_new ()
{
  GstStructure *str = gst_structure_new_empty ("nv-batch-size");
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean
gst_nvquery_is_batch_size (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, "nv-batch-size");
}

void
gst_nvquery_batch_size_set (GstQuery * query, guint batch_size)
{
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);

  if (!gst_nvquery_is_batch_size (query)) {
    return;
  }

  gst_structure_set (str, "batch-size", G_TYPE_UINT, batch_size, NULL);
}

gboolean
gst_nvquery_batch_size_parse (GstQuery * query, guint * batch_size)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (!gst_nvquery_is_batch_size (query)) {
    return FALSE;
  }

  return gst_structure_get_uint (str, "batch-size", batch_size);
}

// ----- Num Surfaces Per Buffer Query handling
GstQuery * gst_nvquery_num_surfaces_per_buffer_new ()
{
  GstStructure *str = gst_structure_new_empty (QUERY_NUM_SURFACES_PER_BUFFER);
  gst_structure_set (str, "batch-size", G_TYPE_UINT, 1, NULL);
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean gst_nvquery_is_num_surfaces_per_buffer (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, QUERY_NUM_SURFACES_PER_BUFFER);
}

void gst_nvquery_num_surfaces_per_buffer_set (GstQuery * query, guint num_surfaces_per_buffers)
{
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);

  if (!gst_nvquery_is_num_surfaces_per_buffer (query)) {
    return;
  }

  gst_structure_set (str, QUERY_NUM_SURFACES_PER_BUFFER, G_TYPE_UINT, num_surfaces_per_buffers, NULL);
}

gboolean gst_nvquery_num_surfaces_per_buffer_parse (GstQuery * query, guint * num_surfaces_per_buffers)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (!gst_nvquery_is_num_surfaces_per_buffer (query)) {
    return FALSE;
  }

  return gst_structure_get_uint (str, QUERY_NUM_SURFACES_PER_BUFFER, num_surfaces_per_buffers);
}
// ----- Num Surfaces Per Buffer Query handling

GstQuery *
gst_nvquery_numStreams_size_new ()
{
  GstStructure *str = gst_structure_new_empty ("nv-numStreams-size");
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean
gst_nvquery_is_numStreams_size (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, "nv-numStreams-size");
}

void
gst_nvquery_numStreams_size_set (GstQuery * query, guint numStreams_size)
{
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);

  if (!gst_nvquery_is_numStreams_size (query)) {
    return;
  }

  gst_structure_set (str, "numStreams-size", G_TYPE_UINT, numStreams_size, NULL);
}

gboolean
gst_nvquery_numStreams_size_parse (GstQuery * query, guint * numStreams_size)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (!gst_nvquery_is_numStreams_size (query)) {
    return FALSE;
  }

  return gst_structure_get_uint (str, "numStreams-size", numStreams_size);
}

GstQuery *
gst_nvquery_resolution_new ()
{
  GstStructure *str = gst_structure_new_empty ("nv-resolution");
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean
gst_nvquery_is_resolution (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, "nv-resolution");
}

void
gst_nvquery_resolution_set (GstQuery * query, guint width, guint height)
{
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);

  if (!gst_nvquery_is_resolution (query)) {
    return;
  }

  gst_structure_set (str, "width", G_TYPE_UINT, width,
      "height", G_TYPE_UINT, height, NULL);
}

gboolean
gst_nvquery_resolution_parse (GstQuery * query, guint * width, guint * height)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (!gst_nvquery_is_resolution (query)) {
    return FALSE;
  }

  return gst_structure_get_uint (str, "width", width) &&
    gst_structure_get_uint (str, "height", height) ;
}

GstQuery *
gst_nvquery_nppstream_new ()
{
  GstStructure *str = gst_structure_new_empty ("nv-nppstream");
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean
gst_nvquery_is_nppstream (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, "nv-nppstream");
}

void
gst_nvquery_nppstream_set (GstQuery * query, gpointer cudastream)
{
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);

  if (!gst_nvquery_is_nppstream (query)) {
    return;
  }

  gst_structure_set (str, "npp-stream", G_TYPE_UINT64, cudastream, NULL);
}

gboolean
gst_nvquery_nppstream_parse (GstQuery * query, gpointer cudastreamptr)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (!gst_nvquery_is_nppstream (query)) {
    return FALSE;
  }

  return gst_structure_get_uint64 (str, "npp-stream", (guint64 *) cudastreamptr);
}

GstQuery * gst_nvquery_ntp_sync_new ()
{
  GstStructure *str = gst_structure_new_empty (QUERY_NTP_SYNC);
  gst_structure_set (str, "batch-size", G_TYPE_UINT, 1, NULL);
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean gst_nvquery_is_ntp_sync (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, QUERY_NTP_SYNC);
}

void gst_nvquery_ntp_sync_set (GstQuery * query, _NtpData *ntp_data)
{
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);

  if (!gst_nvquery_is_ntp_sync (query)) {
    return;
  }

  gst_structure_set (str, "ntp-time-epoch-ns", G_TYPE_UINT64, ntp_data->ntp_time_epoch_ns,
      "frame-timestamp", G_TYPE_UINT64, ntp_data->frame_timestamp,
      "avg-frame-time", G_TYPE_UINT64, ntp_data->avg_frame_time, NULL);
}

gboolean gst_nvquery_ntp_sync_parse (GstQuery * query, _NtpData *ntp_data)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (!gst_nvquery_is_ntp_sync (query)) {
    return FALSE;
  }

  return gst_structure_get (str, "ntp-time-epoch-ns", G_TYPE_UINT64, &ntp_data->ntp_time_epoch_ns,
      "frame-timestamp", G_TYPE_UINT64, &ntp_data->frame_timestamp,
      "avg-frame-time", G_TYPE_UINT64, &ntp_data->avg_frame_time, NULL);
}

GstQuery *
gst_nvquery_stream_caps_new (guint streamId)
{
  GstStructure *str = gst_structure_new_empty ("nv-stream-caps");
  gst_structure_set (str, "streamId", G_TYPE_UINT, streamId, NULL);
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean
gst_nvquery_is_stream_caps (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, "nv-stream-caps");
}

void
gst_nvquery_stream_caps_set (GstQuery * query, GstCaps* caps)
{
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);
  if (!gst_nvquery_is_stream_caps (query)) {
    return;
  }

  GValue caps_val = {0};
  g_value_init(&caps_val, GST_TYPE_CAPS);

  gst_value_set_caps(&caps_val, caps);

  gst_structure_set_value(str, "caps", &caps_val);
}

gboolean
gst_nvquery_stream_caps_parse (GstQuery * query, GstStructure** p_str)
{
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);

  if (!gst_nvquery_is_stream_caps (query)) {
    return FALSE;
  }

  *p_str = str;

  return TRUE;
}

gboolean
gst_nvquery_stream_caps_parse_streamid (GstQuery * query, guint * streamid)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (!gst_nvquery_is_stream_caps (query)) {
    return FALSE;
  }

  return gst_structure_get_uint (str, "streamId", streamid);
}

GstQuery *
gst_nvquery_uri_from_streamid_new (guint streamid)
{
  GstStructure *str = gst_structure_new_empty ("nv-uri-from-streamid");
  gst_structure_set (str, "streamid", G_TYPE_UINT, streamid, NULL);
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean
gst_nvquery_is_uri_from_streamid (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, "nv-uri-from-streamid");
}

gboolean
gst_nvquery_uri_from_streamid_parse (GstQuery * query, const gchar ** uri)
{
  if (!gst_nvquery_is_uri_from_streamid (query))
    return FALSE;

  const GstStructure *str = gst_query_get_structure (query);

  *uri = gst_structure_get_string (str, "uri");

  return TRUE;
}

gboolean
gst_nvquery_uri_from_streamid_parse_streamid (GstQuery * query,
    guint * streamid)
{
  if (!gst_nvquery_is_uri_from_streamid (query))
    return FALSE;

  const GstStructure *str = gst_query_get_structure (query);

  return gst_structure_get (str, "streamid", G_TYPE_UINT, streamid, NULL);
}

void
gst_nvquery_uri_from_streamid_set (GstQuery * query, const gchar * uri)
{
  if (!gst_nvquery_is_uri_from_streamid (query))
    return;
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);
  gst_structure_set (str, "uri", G_TYPE_STRING, uri, NULL);
}


GstQuery *
gst_nvquery_sourceid_new (void)
{
  GstStructure *str = gst_structure_new_empty ("nv-sourceid");
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean
gst_nvquery_is_sourceid (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, "nv-sourceid");
}

gboolean
gst_nvquery_sourceid_parse (GstQuery * query, guint * sourceid)
{
  if (!gst_nvquery_is_sourceid (query))
    return FALSE;

  const GstStructure *str = gst_query_get_structure (query);

  return gst_structure_get (str, "source-id", G_TYPE_UINT, sourceid, NULL);
}

void
gst_nvquery_sourceid_set (GstQuery * query, guint sourceid)
{
  if (!gst_nvquery_is_sourceid (query))
    return;
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);
  gst_structure_set (str, "source-id", G_TYPE_UINT, sourceid, NULL);
}

GstQuery *
gst_nvquery_preprocess_poolsize_new (guint gieId)
{
  GstStructure *str = gst_structure_new_empty ("nv-preprocess-poolsize");
  gst_structure_set (str, "gie-id", G_TYPE_UINT, gieId, NULL);
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean
gst_nvquery_is_preprocess_poolsize (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, "nv-preprocess-poolsize");
}

gboolean
gst_nvquery_preprocess_poolsize_parse (GstQuery * query, guint * bufsize)
{
  if (!gst_nvquery_is_preprocess_poolsize (query))
    return FALSE;

  const GstStructure *str = gst_query_get_structure (query);

  return gst_structure_get (str, "nv-preprocess-poolsize", G_TYPE_UINT, bufsize, NULL);
}

void
gst_nvquery_preprocess_poolsize_set (GstQuery * query, guint bufsize)
{
  if (!gst_nvquery_is_preprocess_poolsize (query))
    return;
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);
  gst_structure_set (str, "nv-preprocess-poolsize", G_TYPE_UINT, bufsize, NULL);
}

gboolean
gst_nvquery_preprocess_poolsize_gieid_parse (GstQuery * query, guint *gieId)
{
  if (!gst_nvquery_is_preprocess_poolsize (query))
    return FALSE;
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);
  return gst_structure_get (str, "gie-id", G_TYPE_UINT, gieId, NULL);
}

gboolean
gst_nvquery_is_update_caps (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, "update-caps");
}

void
gst_nvquery_parse_update_caps (GstQuery *query, guint *stream_index, const GValue *frame_rate)
{
    const GstStructure *str = gst_query_get_structure (query);

    gst_structure_get_uint (str, "stream-id", stream_index);

    frame_rate = gst_structure_get_value (str, "frame-rate");
}

gboolean
gst_nvquery_update_caps_peer_query (GstPad *srcpad, GstStructure *str)
{
    gboolean ret = TRUE;
    GstQuery *caps_query;
    caps_query = gst_query_new_custom (GST_QUERY_CUSTOM, str);
    ret = gst_pad_peer_query (srcpad, caps_query);
    gst_query_unref (caps_query);

    return ret;
}

// ----- Text Embedding Query handling
GstQuery *
gst_nvquery_text_embedding_new (const gchar *text_input, const gchar *model)
{
  GstStructure *str = gst_structure_new_empty (QUERY_TEXT_EMBEDDING);
  gst_structure_set (str, "text", G_TYPE_STRING, text_input,
                          "model", G_TYPE_STRING, model, NULL);
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean
gst_nvquery_is_text_embedding (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, QUERY_TEXT_EMBEDDING);
}

void
gst_nvquery_text_embedding_set (GstQuery * query, const gchar *id,
                                 guint64 created, const gchar *model,
                                 const GValue *data)
{
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);

  if (!gst_nvquery_is_text_embedding (query)) {
    return;
  }

  gst_structure_set (str, "id", G_TYPE_STRING, id,
                          "created", G_TYPE_UINT64, created,
                          "response-model", G_TYPE_STRING, model, NULL);

  if (data != NULL) {
    gst_structure_set_value (str, "data", data);
  }
}

gboolean
gst_nvquery_text_embedding_parse (GstQuery * query, const gchar **id,
                                   guint64 *created, const gchar **model,
                                   const GValue **data)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (!gst_nvquery_is_text_embedding (query)) {
    return FALSE;
  }

  if (id != NULL) {
    *id = gst_structure_get_string (str, "id");
  }

  if (created != NULL) {
    gst_structure_get_uint64 (str, "created", created);
  }

  if (model != NULL) {
    *model = gst_structure_get_string (str, "response-model");
  }

  if (data != NULL) {
    *data = gst_structure_get_value (str,"embeddings");
  }

  return TRUE;
}

gboolean
gst_nvquery_text_embedding_parse_request (GstQuery * query,
                                           const gchar **text_input,
                                           const gchar **model)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (!gst_nvquery_is_text_embedding (query)) {
    return FALSE;
  }

  if (text_input != NULL) {
    *text_input = gst_structure_get_string (str, "text-input");
  }

  if (model != NULL) {
    *model = gst_structure_get_string (str, "model");
  }

  return TRUE;
}
// ----- Text Embedding Query handling

// ----- Image Embedding Query handling
GstQuery *
gst_nvquery_image_embedding_new (const gchar *image_path, const gchar *model,
                                  gboolean has_bbox,
                                  gdouble bbox_left, gdouble bbox_top,
                                  gdouble bbox_width, gdouble bbox_height)
{
  GstStructure *str = gst_structure_new_empty (QUERY_IMAGE_EMBEDDING);
  gst_structure_set (str, "image-path", G_TYPE_STRING, image_path,
                          "model", G_TYPE_STRING, model,
                          "has-bbox", G_TYPE_BOOLEAN, has_bbox,
                          "bbox-left", G_TYPE_DOUBLE, bbox_left,
                          "bbox-top", G_TYPE_DOUBLE, bbox_top,
                          "bbox-width", G_TYPE_DOUBLE, bbox_width,
                          "bbox-height", G_TYPE_DOUBLE, bbox_height, NULL);
  return gst_query_new_custom (GST_QUERY_CUSTOM, str);
}

gboolean
gst_nvquery_is_image_embedding (GstQuery * query)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (GST_QUERY_TYPE (query) != GST_QUERY_CUSTOM)
    return FALSE;

  return gst_structure_has_name (str, QUERY_IMAGE_EMBEDDING);
}

void
gst_nvquery_image_embedding_set (GstQuery * query, const gchar *id,
                                  guint64 created, const gchar *model,
                                  const GValue *data)
{
  GstStructure *str = (GstStructure *) gst_query_get_structure (query);

  if (!gst_nvquery_is_image_embedding (query)) {
    return;
  }

  gst_structure_set (str, "id", G_TYPE_STRING, id,
                          "created", G_TYPE_UINT64, created,
                          "response-model", G_TYPE_STRING, model, NULL);

  if (data != NULL) {
    gst_structure_set_value (str, "data", data);
  }
}

gboolean
gst_nvquery_image_embedding_parse (GstQuery * query, const gchar **id,
                                    guint64 *created, const gchar **model,
                                    const GValue **data)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (!gst_nvquery_is_image_embedding (query)) {
    return FALSE;
  }

  if (id != NULL) {
    *id = gst_structure_get_string (str, "id");
  }

  if (created != NULL) {
    gst_structure_get_uint64 (str, "created", created);
  }

  if (model != NULL) {
    *model = gst_structure_get_string (str, "response-model");
  }

  if (data != NULL) {
    *data = gst_structure_get_value (str, "data");
  }

  return TRUE;
}

gboolean
gst_nvquery_image_embedding_parse_request (GstQuery * query,
                                            const gchar **image_path,
                                            const gchar **model,
                                            gboolean *has_bbox,
                                            gdouble *bbox_left,
                                            gdouble *bbox_top,
                                            gdouble *bbox_width,
                                            gdouble *bbox_height)
{
  const GstStructure *str = gst_query_get_structure (query);

  if (!gst_nvquery_is_image_embedding (query)) {
    return FALSE;
  }

  if (image_path != NULL) {
    *image_path = gst_structure_get_string (str, "image-path");
  }

  if (model != NULL) {
    *model = gst_structure_get_string (str, "model");
  }

  if (has_bbox != NULL) {
    gst_structure_get_boolean (str, "has-bbox", has_bbox);
  }

  if (bbox_left != NULL) {
    gst_structure_get_double (str, "bbox-left", bbox_left);
  }

  if (bbox_top != NULL) {
    gst_structure_get_double (str, "bbox-top", bbox_top);
  }

  if (bbox_width != NULL) {
    gst_structure_get_double (str, "bbox-width", bbox_width);
  }

  if (bbox_height != NULL) {
    gst_structure_get_double (str, "bbox-height", bbox_height);
  }

  return TRUE;
}
// ----- Image Embedding Query handling
