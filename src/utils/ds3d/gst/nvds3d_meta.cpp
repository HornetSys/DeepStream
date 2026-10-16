/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nvds3d_meta.h"

#include "ds3d/common/abi_obj.h"
#include "ds3d/common/common.h"
#include "ds3d/common/hpp/datamap.hpp"
#include "ds3d/common/hpp/obj.hpp"
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

namespace ds3d { namespace impl {

class NvDs3DBufferImpl : public NvDs3DBuffer {
public:
    NvDs3DBufferImpl(abiRefDataMap* datamap, bool takeOwner)
    {
        this->magicID = NVDS3D_BUF_MAGIC_ID;
        DS_ASSERT(datamap);
        if (takeOwner) {
            this->datamap = datamap;
        } else {
            this->datamap = datamap->refCopy();
        }
    }
    ~NvDs3DBufferImpl()
    {
        if (datamap) {
            datamap->destroy();
        }
    }
};

class BatchLocker {
    NvDsBatchMeta* _meta = nullptr;

public:
    BatchLocker(NvDsBatchMeta* meta) : _meta(meta)
    {
        DS_ASSERT(meta);
        nvds_acquire_meta_lock(_meta);
    }
    ~BatchLocker()
    {
        DS_ASSERT(_meta);
        nvds_release_meta_lock(_meta);
    }

private:
    DS3D_DISABLE_CLASS_COPY(BatchLocker);
};

}}  // namespace ds3d::impl

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
ErrCode
NvDs3D_CreateGstBuf(GstBuffer*& outBuf, abiRefDataMap* datamap, bool takeOwner)
{
    DS_ASSERT(datamap);
    impl::NvDs3DBufferImpl* buf3d = new impl::NvDs3DBufferImpl(datamap, takeOwner);
    size_t bufSize = sizeof(impl::NvDs3DBufferImpl);
    GstBuffer* gstBuf = gst_buffer_new_wrapped_full(
        GST_MEMORY_FLAG_READONLY, buf3d, bufSize, 0, bufSize, (void*)buf3d,
        GDestroyNotify(&DeleteTFunc<impl::NvDs3DBufferImpl>));
    DS_ASSERT(gstBuf);
    outBuf = gstBuf;
    return ErrCode::kGood;
}

bool
NvDs3D_IsDs3DBuf(GstBuffer* buf)
{
    GstMapInfo inmap = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(buf, &inmap, GST_MAP_READ)) {
        return false;
    }
    ds3d::UniqPtr<GstMapInfo> safeMapped(
        &inmap, [&inmap, buf](GstMapInfo*) { gst_buffer_unmap(buf, &inmap); });
    if (inmap.size < sizeof(NvDs3DBuffer)) {
        return false;
    }
    NvDs3DBuffer* buf3D = (NvDs3DBuffer*)(inmap.data);
    if (buf3D && buf3D->magicID == NVDS3D_BUF_MAGIC_ID) {
        return true;
    }
    return false;
}

ErrCode
NvDs3D_CreateEmptyBatchMeta(GstBuffer* buf, NvDsBatchMeta*& batchMeta, uint32_t maxBatch)
{
    DS_ASSERT(buf);
    batchMeta = nvds_create_batch_meta(maxBatch);
    if (!batchMeta) {
        return ErrCode::kNvDsMeta;
    }
    batchMeta->num_frames_in_batch = 0;
    NvDsMeta* meta = gst_buffer_add_nvds_meta(
        buf, batchMeta, NULL, nvds_batch_meta_copy_func, nvds_batch_meta_release_func);

    DS_ASSERT(meta);
    meta->meta_type = NVDS_BATCH_GST_META;
    return ErrCode::kGood;
}

static void
releaseDataMapMeta(gpointer data, gpointer user_data)
{
    NvDsUserMeta* user_meta = (NvDsUserMeta*)data;
    abiRefDataMap* datamap = (abiRefDataMap*)user_meta->user_meta_data;
    if (datamap) {
        datamap->destroy();
    }
}

static gpointer
copyDataMapMeta(gpointer data, gpointer user_data)
{
    NvDsUserMeta* user_meta = (NvDsUserMeta*)data;
    abiRefDataMap* datamap = (abiRefDataMap*)user_meta->user_meta_data;
    if (!datamap) {
        return nullptr;
    }
    return (abiRefDataMap*)datamap->refCopy();
}


ErrCode
NvDs3D_BatchMeta_SetDataMapAsMeta(NvDsBatchMeta* batchMeta, const abiRefDataMap* datamap)
{
    DS_ASSERT(batchMeta);
    DS_ASSERT(datamap);
    NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batchMeta);
    abiRefDataMap* newData = datamap->refCopy();
    DS_ASSERT(newData);
    user_meta->user_meta_data = newData;
    user_meta->base_meta.meta_type = (NvDsMetaType)NVDS_3D_DATAMAP_META_TYPE;
    user_meta->base_meta.release_func = releaseDataMapMeta;
    user_meta->base_meta.copy_func = copyDataMapMeta;

    nvds_add_user_meta_to_batch(batchMeta, user_meta);
    return ErrCode::kGood;
}

ErrCode
NvDs3D_Find1stDataMap(GstBuffer* buf, const abiRefDataMap*& datamap)
{
    if (NvDs3D_IsDs3DBuf(buf)) {
        GstMapInfo inmap = GST_MAP_INFO_INIT;
        if (!gst_buffer_map(buf, &inmap, GST_MAP_READ)) {
            return ErrCode::kMem;
        }
        UniqPtr<GstMapInfo> safeMapped(
            &inmap, [&inmap, buf](GstMapInfo*) { gst_buffer_unmap(buf, &inmap); });

        NvDs3DBuffer* buf3D = (NvDs3DBuffer*)(inmap.data);
        DS_ASSERT(buf3D && buf3D->magicID == NVDS3D_BUF_MAGIC_ID);
        datamap = buf3D->datamap;
        return ErrCode::kGood;
    } else {
        // If the buffer is not DS3D buffer, then try to get from batch meta.
        NvDsBatchMeta* batchMeta = gst_buffer_get_nvds_batch_meta(buf);
        if (!batchMeta) {
            return ErrCode::kNvDsMeta;
        }
        NvDsMetaList* iUserMeta = nullptr;
        NvDsUserMeta* userMeta = nullptr;
        impl::BatchLocker locker(batchMeta);
        for (iUserMeta = batchMeta->batch_user_meta_list; iUserMeta != NULL;
             iUserMeta = iUserMeta->next) {
            userMeta = (NvDsUserMeta*)(iUserMeta->data);
            if (userMeta->base_meta.meta_type == NVDS_3D_DATAMAP_META_TYPE) {
                abiRefDataMap* data = static_cast<abiRefDataMap*>(userMeta->user_meta_data);
                DS_ASSERT(data);
                datamap = data;
                return ErrCode::kGood;
            }
        }
    }
    return ErrCode::kNotFound;
}

ErrCode
NvDs3D_UpdateDataMap(GstBuffer* buf, const ds3d::abiRefDataMap* datamap)
{
    DS_ASSERT(datamap);
    if (NvDs3D_IsDs3DBuf(buf)) {
        GstMapInfo inmap = GST_MAP_INFO_INIT;
        if (!gst_buffer_map(buf, &inmap, GST_MAP_READ)) {
            return ErrCode::kMem;
        }
        UniqPtr<GstMapInfo> safeMapped(
            &inmap, [&inmap, buf](GstMapInfo*) { gst_buffer_unmap(buf, &inmap); });

        NvDs3DBuffer* buf3D = (NvDs3DBuffer*)(inmap.data);
        DS_ASSERT(buf3D && buf3D->magicID == NVDS3D_BUF_MAGIC_ID);
        GuardDataMap release(buf3D->datamap, true);
        buf3D->datamap = datamap->refCopy();
        return ErrCode::kGood;
    } else {
        ErrCode c = ErrCode::kGood;
        // If the buffer is not DS3D buffer, then try to get from batch meta.
        NvDsBatchMeta* batchMeta = gst_buffer_get_nvds_batch_meta(buf);
        if (!batchMeta) {
            c = NvDs3D_CreateEmptyBatchMeta(buf, batchMeta, 16);
            DS3D_ERROR_RETURN(c, "create empty batch meata into ds3d failed");
            ;
        }
        NvDsMetaList* iUserMeta = nullptr;
        NvDsUserMeta* userMeta = nullptr;
        impl::BatchLocker locker(batchMeta);
        for (iUserMeta = batchMeta->batch_user_meta_list; iUserMeta != NULL;
             iUserMeta = iUserMeta->next) {
            userMeta = (NvDsUserMeta*)(iUserMeta->data);
            if (userMeta->base_meta.meta_type == NVDS_3D_DATAMAP_META_TYPE) {
                GuardDataMap release(static_cast<abiRefDataMap*>(userMeta->user_meta_data), true);
                userMeta->user_meta_data = (void*)datamap->refCopy();
                return ErrCode::kGood;
            }
        }
        c = NvDs3D_BatchMeta_SetDataMapAsMeta(batchMeta, datamap);
        DS3D_ERROR_RETURN(c, "NvDs3D_BatchMeta_SetDataMapAsMeta failed");
        return ErrCode::kGood;
    }
    return ErrCode::kNotFound;
}

DS3D_EXTERN_C_END
