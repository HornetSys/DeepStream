/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ds3d/common/impl/impl_frames.h"
#include "ds3d/common/ds3d_analysis_datatype.h"
#include "ds3d/common/idatatype.h"
#include <gst/gst.h>
#include "nvbufsurface.h"
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h"
#include "video_bridge_impl.h"

#include <unistd.h>

namespace ds3d { namespace impl { namespace bridge {

class InProcess {
    volatile bool& _inFlag;

public:
    InProcess(volatile bool& flag) : _inFlag(flag) { _inFlag = flag; }
    ~InProcess() { release(); }
    void release() { _inFlag = false; }
};

VideoBridge2d3d::~VideoBridge2d3d()
{
    stop_i();
}

ErrCode
VideoBridge2d3d::processImpl(
    const struct VideoBridge2dInput* inputData, OnGuardDataCBImpl outputDataCb, OnGuardBridgeDataCBImpl inputConsumedCb)
{
    LOG_DEBUG("databridge is starting to bridge datamap");
    std::unique_lock<ImplMutex> locker(mutex());
    InProcess inProcess(_inProcess);

    auto destructorFn = [inputConsumedCb](struct VideoBridge2dInput* in) {
        inputConsumedCb(ErrCode::kGood, in);
    };
    ShrdPtr<VideoBridge2dInput> inputPtr(const_cast<VideoBridge2dInput*>(inputData), destructorFn);

    /** Create datamap from inputData */
    GuardDataMap datamap;
    datamap.reset(NvDs3d_CreateDataHashMap());

    TimeStamp ts{0};
    ts.t0 = GST_BUFFER_TIMESTAMP(inputPtr->buffer);
    DS3D_ERROR_RETURN(datamap.setData(kTimeStamp, ts), "failed to set bridge2dData: %s", kTimeStamp);

    std::string bridgeKey(kVideoBridge2dData);
    DS3D_ERROR_RETURN(
        datamap.setPtrData(bridgeKey, inputPtr), "failed to set bridge2dData: %s",
        bridgeKey.c_str());

    if (_config.surfaceToImage) {
        NvBufSurface* surface = inputPtr->surfaceBatchBuffer;
        DS3D_FAILED_RETURN(
            surface && surface->numFilled, ErrCode::kNotFound,
            "No surface is found in 2d_to_3d_bridge");
        DS3D_FAILED_RETURN(
            surface->memType == NVBUF_MEM_DEFAULT || surface->memType == NVBUF_MEM_CUDA_DEVICE,
            ErrCode::kUnsupported, "surface type is not support in 2d_to_3d_bridge");
        const NvBufSurfaceParams* parms = surface->surfaceList;
        DS3D_FAILED_RETURN(
            parms->colorFormat == NVBUF_COLOR_FORMAT_RGBA, ErrCode::kUnsupported,
            "input surface color format must be RGBA, other formats are not supported");
        DS_ASSERT(parms);
        Frame2DPlane plane = {
            parms->width, parms->height, parms->planeParams.pitch[0], sizeof(vec4b), 0};
        Frame2DGuard imageFrame = impl::Wrap2DFrame<uint8_t, FrameType::kColorRGBA>(
            parms->dataPtr, {plane}, parms->dataSize, MemType::kGpuCuda, surface->gpuId,
            [inputPtr](void*) {});
        DS_ASSERT(imageFrame);
        /// add main surface without index
        DS3D_ERROR_RETURN(
            datamap.setGuardData(_config.outputImageKey, imageFrame),
            "Failed to set output frame: %s", _config.outputImageKey.c_str());

        /// add multi-surface with index number, this is for multi-review features
        for (uint32_t idx = 0; idx < surface->numFilled; ++idx) {
            const NvBufSurfaceParams& pIdx = parms[idx];
            uint32_t streamId = idx;
            if (inputPtr->batchMeta) {
                NvDsFrameMeta* frameMeta =
                    (NvDsFrameMeta*)g_list_nth_data(inputPtr->batchMeta->frame_meta_list, idx);
                DS3D_FAILED_RETURN(
                    frameMeta, ErrCode::kUnknown, "video_bridge get frame meta failed with idx:%u",
                    idx);
                streamId = frameMeta->pad_index;
            }
            std::string keyIdx = _config.outputImageKey + "_" + std::to_string(streamId);
            Frame2DPlane planeIdx = {
                pIdx.width, pIdx.height, pIdx.planeParams.pitch[0], sizeof(vec4b), 0};
            Frame2DGuard imageIdx = impl::Wrap2DFrame<uint8_t, FrameType::kColorRGBA>(
                pIdx.dataPtr, {planeIdx}, pIdx.dataSize, MemType::kGpuCuda, surface->gpuId,
                [inputPtr](void*) {});
            DS_ASSERT(imageIdx);
            DS3D_ERROR_RETURN(
                datamap.setGuardData(keyIdx, imageIdx), "Failed to set output frame: %s",
                keyIdx.c_str());
        }
    }

    if (inputPtr->batchMeta) {
        DS3D_ERROR_RETURN(
            extractBatchMeta(inputPtr->batchMeta, datamap, inputPtr),
            "Extract batch_meta into datamap failed");
    }

    inProcess.release();
    locker.unlock();
    LOG_DEBUG("videobridge output:");
    datamap.printDebug();
    outputDataCb(ErrCode::kGood, datamap);

    return ErrCode::kGood;
}

ErrCode
VideoBridge2d3d::extractBatchMeta(NvDsBatchMeta* batchMeta, GuardDataMap& datamap,
                                  ShrdPtr<VideoBridge2dInput>& inputPtr)
{
    const NvDsMetaList * frame = nullptr;
    const NvDsMetaList * obj = NULL;
    const NvDsObjectMeta *objMeta = nullptr;
    const NvDsMetaList *userMetaList = nullptr;
    std::vector<Object2DBbox> objBboxVec;
    objBboxVec.reserve(16);

    g_rec_mutex_lock (&batchMeta->meta_mutex);
    for (frame = batchMeta->frame_meta_list; frame != nullptr; frame = frame->next) {
        const NvDsFrameMeta *frameMeta = (const NvDsFrameMeta *) (frame->data);
        for (obj = frameMeta->obj_meta_list; obj != nullptr;
                obj = obj->next) {
            objMeta = (const NvDsObjectMeta *) (obj->data);
            DS_ASSERT(objMeta);
            const NvOSD_RectParams& rect = objMeta->rect_params;

            Object2DBbox bbox;
            bbox.cid = objMeta->class_id;
            bbox.sourceId = (int)frameMeta->pad_index;
            bbox.score = objMeta->confidence;
            bbox.centerX = rect.left + (rect.width / 2.0f);
            bbox.centerY = rect.top + (rect.height / 2.0f);
            bbox.dx = rect.width;
            bbox.dy = rect.height;
            snprintf(bbox.labels, sizeof(bbox.labels), "%s", objMeta->obj_label);
            const NvOSD_ColorParams& c = rect.border_color;
            bbox.bboxColor = {{(float)c.red, (float)c.green, (float)c.blue, (float)c.alpha}};
            objBboxVec.emplace_back(bbox);
        }

        for (userMetaList = batchMeta->batch_user_meta_list; userMetaList != nullptr;
                userMetaList = userMetaList->next) {
            const NvDsUserMeta *userMeta = (NvDsUserMeta *) (userMetaList->data);
            if (userMeta->base_meta.meta_type == NVDS_PREPROCESS_BATCH_META) {
                auto preprocMeta = (GstNvDsPreProcessBatchMeta *) userMeta->user_meta_data;
                if (!preprocMeta->tensor_meta) {
                    LOG_DEBUG("tensor_meta is missing in GstNvDsPreProcessBatchMeta");
                    break;
                }

                int i = 0;
                Shape preprocMetaShape = { 0 };
                preprocMetaShape.numDims = preprocMeta->tensor_meta->tensor_shape.size();
                for (auto d : preprocMeta->tensor_meta->tensor_shape) {
                    preprocMetaShape.d[i++] = d;
                }
                FrameGuard preprocTensorFrame = impl::WrapFrame<uint8_t, FrameType::kCustom>(
                                            preprocMeta->tensor_meta->raw_tensor_buffer,
                                            preprocMeta->tensor_meta->buffer_size,
                                            preprocMetaShape, MemType::kGpuCuda,
                                            preprocMeta->tensor_meta->gpu_id, [inputPtr](void*) {});
                DS3D_ERROR_RETURN(
                    datamap.setGuardData(_config.outputVideoPreProcTensorKey, preprocTensorFrame),
                        "insert video preproc tensor (key: %s) into datamap failed.", _config.outputVideoPreProcTensorKey.c_str());
                break;
            }
        }
    }
    g_rec_mutex_unlock (&batchMeta->meta_mutex);

    // attach object2DBbox into datamap
    if (objBboxVec.size()) {
        void *bufPtr = &objBboxVec[0];
        size_t bytes = objBboxVec.size() * sizeof(Object2DBbox);
        Shape shape{2, {(int)objBboxVec.size(), sizeof(Object2DBbox)}};
        FrameGuard obj2DFrame = impl::WrapFrame<uint8_t, FrameType::kCustom>(bufPtr, bytes, shape,
                    MemType::kCpu, 0, [keep = std::move(objBboxVec)](void*) {});
        DS3D_ERROR_RETURN(datamap.setGuardData(_config.outputObject2DKey, obj2DFrame),
            "insert object 2D bbox (key: %s) into datamap failed.", _config.outputObject2DKey.c_str());
    }

    return ErrCode::kGood;
}

void
VideoBridge2d3d::timeoutWaitProcess()
{
    int times = 0;
    while (_inProcess && times++ < 5) {
        usleep(1000 * 1000);
    }
}

ErrCode
VideoBridge2d3d::stopImpl()
{
    LOG_INFO("databridge is closing");
    std::unique_lock<ImplMutex> locker(mutex());
    timeoutWaitProcess();

    return ErrCode::kGood;
}

ErrCode
VideoBridge2d3d::flushImpl()
{
    LOG_INFO("databridge is flushing");

    std::unique_lock<ImplMutex> locker(mutex());
    timeoutWaitProcess();
    return ErrCode::kGood;
}

ErrCode
VideoBridge2d3d::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("video 2d->3d databridge is starting");
    std::unique_lock<ImplMutex> locker(mutex());
    ErrCode code =
        config::CatchYamlCall([&, this]() { return parseConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse databridge video 2d->3d config: %s failed", path.c_str());
    setInputCaps(_config.compConfig.gstInCaps);
    setOutputCaps(_config.compConfig.gstOutCaps);

    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::bridge

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataBridge*
createVideoBridge2d3d()
{
    return NewAbiRef<abiDataBridge>(new impl::bridge::VideoBridge2d3d);
}

DS3D_EXTERN_C_END
