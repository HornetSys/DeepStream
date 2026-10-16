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

#include "gst-nvmessage.h"

#define STREAM_EOS_STRUCT_NAME "stream-eos"

#define CHECK_MESSAGE_TYPE(message,type) do { \
  const GstStructure *str; \
  if (GST_MESSAGE_TYPE (message) != GST_MESSAGE_ELEMENT) \
    return FALSE; \
  str = gst_message_get_structure (message); \
  return (str != NULL) && gst_structure_has_name (str,type); \
} while (0)

GstMessage *
gst_nvmessage_new_stream_eos (GstObject * obj, guint eos_stream_id)
{
  GstStructure *str =
      gst_structure_new (STREAM_EOS_STRUCT_NAME, "stream-id", G_TYPE_UINT,
      eos_stream_id, NULL);
  GstMessage *message = gst_message_new_custom (GST_MESSAGE_ELEMENT, obj, str);

  return message;
}

gboolean
gst_nvmessage_is_stream_eos (GstMessage * message)
{
  CHECK_MESSAGE_TYPE (message, STREAM_EOS_STRUCT_NAME);
}

gboolean
gst_nvmessage_parse_stream_eos (GstMessage * message,
    guint * eos_stream_id)
{
  const GstStructure *str;

  if (!gst_nvmessage_is_stream_eos (message))
    return FALSE;

  str = gst_message_get_structure (message);
  return gst_structure_get_uint (str, "stream-id", eos_stream_id);
}
