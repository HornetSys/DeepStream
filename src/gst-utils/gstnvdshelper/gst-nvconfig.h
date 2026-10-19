/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file
 * <b>NVIDIA GStreamer DeepStream: Custom Configs</b>
 *
 * @b Description: This file specifies the NVIDIA DeepStream config parameters,
 * used in Gstreamer custom events to individual sources which
 * are batched together by Gst-nvstreammux.
 *
 */

/**
 * @defgroup  gstreamer_nvevent  config: Custom Events API
 *
 * Specifies GStreamer custom event functions, used in Gstreamer custom events
 * to individual sources which are batched together by Gst-nvstreammux.
 *
 * @ingroup
 * @{
 */

#ifndef __GST_NVCONFIG_H__
#define __GST_NVCONFIG_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Defines Roi structure for GST_NVEVENT_ROI_UPDATE custom event */
typedef struct RoiDimension{
  gchar roi_id[128];
  guint left;
  guint top;
  guint width;
  guint height;
} RoiDimension;

#ifdef __cplusplus
}
#endif

#endif

/** @} */
