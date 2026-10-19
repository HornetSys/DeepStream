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

#include <stdatomic.h>
#include "gst-nvevent.h"

/** Maximum length of the string buffer */
#define MAX_STR_LEN 64

/** Macro to create an empty GstStructure to be used with the custom events.
 * The structure name is created by concatenating the event name with the
 * source ID and a counter so that the names are unique. This ensures that
 * the events from different pads do not overwrite each other in the sticky
 * events array in case of a leaky queue in the pipeline.
 */
#define CREATE_EMPTY_GST_STRUCT(p_struct, name, id)  \
  static atomic_int ctr = 0; \
  char buffer_name[MAX_STR_LEN]; \
  g_snprintf(buffer_name, MAX_STR_LEN, name "-%d-%d", id, ctr++); \
  GstStructure *p_struct = gst_structure_new_empty (buffer_name); \


GstEvent *
gst_nvevent_new_pad_added (guint source_id)
{
  CREATE_EMPTY_GST_STRUCT(str, "nv-pad-added", source_id);

  gst_structure_set (str, "source-id", G_TYPE_UINT, source_id, NULL);

  return gst_event_new_custom (GST_NVEVENT_PAD_ADDED, str);
}

GstEvent *
gst_nvevent_new_pad_deleted (guint source_id)
{
  CREATE_EMPTY_GST_STRUCT(str, "nv-pad-deleted", source_id);

  gst_structure_set (str, "source-id", G_TYPE_UINT, source_id, NULL);

  return gst_event_new_custom (GST_NVEVENT_PAD_DELETED, str);
}

GstEvent *
gst_nvevent_new_stream_eos (guint source_id)
{
  CREATE_EMPTY_GST_STRUCT(str, "nv-stream-eos", source_id);

  gst_structure_set (str, "source-id", G_TYPE_UINT, source_id,  NULL);

  return gst_event_new_custom (GST_NVEVENT_STREAM_EOS, str);
}

GstEvent *
gst_nvevent_new_stream_segment (guint source_id, GstSegment *segment)
{
  CREATE_EMPTY_GST_STRUCT(str, "nv-stream-segment", source_id);

  gst_structure_set (str, "source-id", G_TYPE_UINT, source_id,
      "segment", GST_TYPE_SEGMENT, segment, NULL);

  return gst_event_new_custom (GST_NVEVENT_STREAM_SEGMENT, str);
}

GstEvent * gst_nvevent_new_stream_start (guint source_id, gchar* stream_id)
{
  CREATE_EMPTY_GST_STRUCT(str, "nv-stream-start", source_id);

  gst_structure_set (str, "source-id", G_TYPE_UINT, source_id,
        "stream-id", G_TYPE_STRING, stream_id, NULL);

  return gst_event_new_custom (GST_NVEVENT_STREAM_START, str);
}

GstEvent * gst_nvevent_new_update_caps (guint pad_id, guint width_val,
		guint height_val, GstStructure *caps_str, gchar* stream_id_str, gboolean is_audio )
{
  CREATE_EMPTY_GST_STRUCT(str, "update-caps", pad_id);

  if(is_audio==TRUE){
    gst_structure_set (str, "stream-id", G_TYPE_UINT, pad_id,
        "stream-id-str", G_TYPE_STRING, stream_id_str, NULL);
  }else {
    gst_structure_set (str, "stream-id", G_TYPE_UINT, pad_id,"stream-id-str", G_TYPE_STRING, stream_id_str,
        "width-val", G_TYPE_UINT, width_val, "height-val", G_TYPE_UINT, height_val, NULL);
  }

  if (gst_structure_has_field (caps_str, "framerate")) {
	  gst_structure_set_value (str, "frame-rate",
			  gst_structure_get_value (caps_str, "framerate"));
  }

  return gst_event_new_custom (GST_NVEVENT_UPDATE_CAPS, str);
}

GstEvent *
gst_nvevent_new_stream_reset (guint source_id)
{
  CREATE_EMPTY_GST_STRUCT(str, "nv-stream-reset", source_id);

  gst_structure_set (str, "source-id", G_TYPE_UINT, source_id,  NULL);

  return gst_event_new_custom (GST_NVEVENT_STREAM_RESET, str);
}

void
gst_nvevent_parse_pad_added (GstEvent * event, guint * source_id)
{
  if((GstEventType) GST_NVEVENT_PAD_ADDED == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get_uint (str, "source-id", source_id);
  }
}

void
gst_nvevent_parse_pad_deleted (GstEvent * event, guint * source_id)
{
  if((GstEventType) GST_NVEVENT_PAD_DELETED == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get_uint (str, "source-id", source_id);
  }
}

void
gst_nvevent_parse_stream_eos (GstEvent * event, guint * source_id)
{
  if((GstEventType) GST_NVEVENT_STREAM_EOS == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "source-id", G_TYPE_UINT, source_id, NULL);
  }
}

void
gst_nvevent_parse_stream_segment (GstEvent * event, guint * source_id,
    GstSegment **segment)
{
  if((GstEventType) GST_NVEVENT_STREAM_SEGMENT == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "source-id", G_TYPE_UINT, source_id,
                            "segment", GST_TYPE_SEGMENT, segment, NULL);
  }
}

void
gst_nvevent_parse_stream_reset (GstEvent * event, guint * source_id)
{
  if((GstEventType) GST_NVEVENT_STREAM_RESET == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "source-id", G_TYPE_UINT, source_id, NULL);
  }
}

void
gst_nvevent_parse_stream_start (GstEvent * event, guint * source_id, gchar** stream_id)
{
  if((GstEventType) GST_NVEVENT_STREAM_START == GST_EVENT_TYPE (event)) {

    const GstStructure *str = gst_event_get_structure (event);

    gst_structure_get (str, "source-id", G_TYPE_UINT, source_id,
        "stream-id", G_TYPE_STRING, stream_id, NULL);
  }
}
