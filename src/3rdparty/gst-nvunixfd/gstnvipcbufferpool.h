/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef GSTNVIPCBUFFERPOOL_H_
#define GSTNVIPCBUFFERPOOL_H_

#include <gst/gst.h>
#include "nvbufsurface.h"

G_BEGIN_DECLS

typedef struct _GstNvIpcBufferPool GstNvIpcBufferPool;
typedef struct _GstNvIpcBufferPoolClass GstNvIpcBufferPoolClass;
typedef struct _GstNvIpcBufferPoolPrivate GstNvIpcBufferPoolPrivate;

#define GST_TYPE_NVIPC_BUFFER_POOL      (gst_nvipc_buffer_pool_get_type())
#define GST_IS_NVIPC_BUFFER_POOL(obj)   (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_NVIPC_BUFFER_POOL))
#define GST_NVIPC_BUFFER_POOL(obj)      (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_NVIPC_BUFFER_POOL, GstNvIpcBufferPool))
#define GST_NVIPC_BUFFER_POOL_CAST(obj) ((GstNvIpcBufferPool*)(obj))

#define GST_NVIPC_MEMORY_TYPE "nvipc"
#define GST_BUFFER_POOL_OPTION_NVIPC_META "GstBufferPoolOptionNvIpcMeta"
#define NV_GST_BUFFER_POOL_MAX_SIZE 8

struct _GstNvIpcBufferPool
{
  GstBufferPool bufferpool;

  GstNvIpcBufferPoolPrivate *priv;
  guint num_queued;
};

struct _GstNvIpcBufferPoolClass
{
  GstBufferPoolClass parent_class;
};

GType gst_nvipc_buffer_pool_get_type (void);

GstBufferPool* gst_nvipc_buffer_pool_new (void);

G_END_DECLS

#endif /* GSTNVIPCBUFFERPOOL_H_ */
