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

#include "alignment_impl.h"
#include "ds3d/common/impl/impl_frames.h"
#include "ds3d/glprocess/gl_helper.h"

#include <unistd.h>

namespace ds3d { namespace impl { namespace filter {

namespace {
ErrCode
reserveMem(std::shared_ptr<BufferPool<MemPtr>>& pool, size_t memSize, uint32_t count, const std::string& name)
{
    pool = std::make_shared<BufferPool<MemPtr>>(name);
    DS_ASSERT(pool);
    for (uint32_t i = 0; i < count; ++i) {
        MemPtr ptr = CpuMemBuf::CreateBuf(memSize);
        pool->setBuffer(std::move(ptr));
    }
    DS3D_FAILED_RETURN(pool->size() == count, ErrCode::kMem, "allocate CPU memory failed on %s", name.c_str());
    return ErrCode::kGood;
}

}  // namespace

AlignmentFilter::~AlignmentFilter()
{
    stop_i();
}

template <typename IN, typename OUT, int InSize = sizeof(IN) / sizeof(float), int OutSize = sizeof(OUT) / sizeof(float)>
static void
alignData(
    FrameGuard& lidarFrame, std::shared_ptr<MemData>& dataLidarAlignedXYZI, glm::mat4& transform, size_t numPoints)
{
    IN* inputLidarPoints = (IN*)lidarFrame->base();
    OUT* outputLidarPoints = (OUT*)dataLidarAlignedXYZI->data;
    std::vector<vec4f> transformV = ds3d::gl::glmMat4ToDS(transform);
    for (size_t p = 0; p < numPoints; p++) {
        // transformV, inputLidarPoints, outputLidarPoints are column-major
        float w = (InSize == 4) ? inputLidarPoints->data[3] : 1.0;
        for (size_t i = 0; i < 3; i++) {
            outputLidarPoints->data[i] = transformV[0].data[i] * inputLidarPoints->x +
                                         transformV[1].data[i] * inputLidarPoints->y +
                                         transformV[2].data[i] * inputLidarPoints->z + transformV[3].data[i];
        }
        if (OutSize == 4) {
            // copy intensity from input
            outputLidarPoints->data[3] = w;
        }
        inputLidarPoints++;
        outputLidarPoints++;
    }
}

ErrCode
AlignmentFilter::addConfigDataTo(GuardDataMap& datamap)
{
    uint32_t numCam =
        std::min<uint32_t>(_config.outputCamIntrMatKeys.size(), _config.camIntrinsicMat.size());
    for (uint32_t i = 0; i < numCam; ++i) {
        DS3D_FAILED_RETURN(
            isGood(datamap.setData(_config.outputCamIntrParmKeys[i], _config.camIntrinsicParm[i])),
            ErrCode::kMem, "adding camIntrinsicParm[%d] to datamap failed", i);

        DS3D_FAILED_RETURN(
            isGood(datamap.setData(_config.outputCamIntrMatKeys[i], _config.camIntrinsicMat[i])),
            ErrCode::kMem, "adding camIntrinsicMat[%d] to datamap failed", i);
    }

    uint32_t numExt = std::min<uint32_t>(
        _config.outputLidarToCamExtrMatKeys.size(), _config.lidarToCamExtrinsicMat.size());
    for (uint32_t i = 0; i < numExt; ++i) {
        DS3D_FAILED_RETURN(
            isGood(datamap.setData(
                _config.outputLidarToCamExtrMatKeys[i], _config.lidarToCamExtrinsicMat[i])),
            ErrCode::kMem, "adding lidarToCamExtrinsicMat[%d] to datamap failed", i);
    }
    return ErrCode::kGood;
}

ErrCode
AlignmentFilter::copyFromRetentionTo(GuardDataMap& datamap)
{
    for (size_t i = 0; i < _config.lidarAlignedKey.size(); i++) {
        if (!datamap.hasData(_config.lidarAlignedKey[i])) {
            datamap.copy(retentionDataMap, _config.lidarAlignedKey[i]);
        }
    }
    for (size_t i = 0; i < _config.camInputKey.size(); i++) {
        if (!datamap.hasData(_config.camInputKey[i])) {
            datamap.copy(retentionDataMap, _config.camInputKey[i]);
        }
    }
    for (size_t i = 0; i < _config.lidarInputKey.size(); i++) {
        if (!datamap.hasData(_config.lidarInputKey[i])) {
            datamap.copy(retentionDataMap, _config.lidarInputKey[i]);
        }
    }
    return ErrCode::kGood;
}

/** Alignment Input and Output
 * Input:
 * Key                : Value
 ***************************************************************************
 * kLidarXYZI         : nX3
 * kTimeStamp         : time
 * kVideoBridge2dData : video
 ***************************************************************************
 * Output = Input + Below.
 * Key                : Value
 ***************************************************************************
 * kLidarXYZIAlignedKey : nX3 Aligned = VTC * (P *) input
 * transformMatrix: P X VTC
 * cam_intrinsic_parm: IntrinsicsParam
 * cam_intrinsic_mat: TransformMatrix
 * lidar_to_cam_extrinsic_mat: TransformMatrix
 ***************************************************************************
 */
ErrCode
AlignmentFilter::processImpl(GuardDataMap datamap, OnGuardDataCBImpl outputDataCb, OnGuardDataCBImpl inputConsumedCb)
{
    LOG_DEBUG("lidar alignment datafilter is starting to filter datamap");
    std::unique_lock<ImplMutex> locker(mutex());
    FrameGuard lidarFrame;
    LOG_DEBUG("alignment filter input:");
    datamap.printDebug();

    for (size_t i = 0; i < _config.lidarInputKey.size() && !_config.disableLidarDataAlign; i++) {
        std::string filterInputDatamapKey = _config.lidarInputKey[i];

        if (datamap.getGuardData(filterInputDatamapKey, lidarFrame) != ErrCode::kGood) {
            LOG_DEBUG("No lidar data found in datamap from alignment filter\n");
            continue;
        }

        /** Get mat4 from lidarFrame */
        DS3D_FAILED_RETURN(
            (lidarFrame->frameType() == FrameType::kPointXYZ || lidarFrame->frameType() == FrameType::kLidarXYZI) &&
                lidarFrame->dataType() == DataType::kFp32,
            ErrCode::kUnsupported, "lidar alignment only support PointXYZ|LidarXYZI with FP32");

        Shape pShape = lidarFrame->shape();  // N x 3 | N x 4
        DS_ASSERT(pShape.numDims);
        // XYZ | XYZI
        uint32_t frameEleSize = pShape.d[pShape.numDims - 1];
        size_t numPoints = std::min((uint32_t)(ShapeSize(pShape) / frameEleSize), _config.maxPoints);

        auto& camIntrinsicMat = _config.camIntrinsicMat[0];
        std::vector<vec4f> camIntrinsicMat4(
            std::begin(camIntrinsicMat.matrix), std::end(camIntrinsicMat.matrix));
        glm::mat4 camIntrinsicGLMMat = ds3d::gl::dsMat4ToGlm(camIntrinsicMat4);

        auto& lidarToCamExtrinsicMat = _config.lidarToCamExtrinsicMat[0];
        std::vector<vec4f> lidarToCamExtrinsicMat4(
            std::begin(lidarToCamExtrinsicMat.matrix), std::end(lidarToCamExtrinsicMat.matrix));
        glm::mat4 lidarToCamExtrinsicGLMMat = ds3d::gl::dsMat4ToGlm(lidarToCamExtrinsicMat4);

        std::shared_ptr<MemData> dataLidarAlignedXYZI = _alignedOpPool->acquireBuffer();
        glm::mat4 transform;
        if (_config.alignToIntrinsic) {
            transform = camIntrinsicGLMMat * lidarToCamExtrinsicGLMMat;
        } else {
            transform = lidarToCamExtrinsicGLMMat;
        }

        if (frameEleSize == 4 && _config.lidarElemSize == 4)
            alignData<vec4f, vec4f>(lidarFrame, dataLidarAlignedXYZI, transform, numPoints);
        if (frameEleSize == 4 && _config.lidarElemSize == 3)
            alignData<vec4f, vec3f>(lidarFrame, dataLidarAlignedXYZI, transform, numPoints);
        if (frameEleSize == 3 && _config.lidarElemSize == 3)
            alignData<vec3f, vec3f>(lidarFrame, dataLidarAlignedXYZI, transform, numPoints);
        if (frameEleSize == 3 && _config.lidarElemSize == 4)
            alignData<vec3f, vec4f>(lidarFrame, dataLidarAlignedXYZI, transform, numPoints);

        void* pointPtr = dataLidarAlignedXYZI->data;
        uint32_t pointNum = std::min(numPoints, dataLidarAlignedXYZI->byteSize);
        FrameGuard pointXyzi;
        if (_config.lidarElemSize == 3) {
            pointXyzi = impl::wrapPointXYZFrame<float>(
                pointPtr, pointNum, MemType::kCpu, 0, [data = std::move(dataLidarAlignedXYZI)](void*) {});
        } else {
            // assume 4
            pointXyzi = impl::wrapLidarXYZIFrame<float>(
                pointPtr, pointNum, MemType::kCpu, 0, [data = std::move(dataLidarAlignedXYZI)](void*) {});
        }
        DS3D_FAILED_RETURN(
            isGood(datamap.setGuardData(_config.lidarAlignedKey[i], pointXyzi)), ErrCode::kMem,
            "adding lidar aligned data output to datamap failed");
    }

    DS3D_ERROR_RETURN(addConfigDataTo(datamap), "adding config matrices into datamap failed");

    DS3D_ERROR_RETURN(retentionDataMap.copy(datamap), "copy output datamap to retention failed");

    DS3D_ERROR_RETURN(copyFromRetentionTo(datamap), "copy retention to output datamap failed");

    locker.unlock();

    LOG_DEBUG("alignment filter output:");
    datamap.printDebug();

    inputConsumedCb(ErrCode::kGood, datamap);
    outputDataCb(ErrCode::kGood, datamap);

    return ErrCode::kGood;
}


ErrCode
AlignmentFilter::stopImpl()
{
    LOG_INFO("lidar alignment datafilter is closing");
    std::unique_lock<ImplMutex> locker(mutex());
    retentionDataMap.reset();
    return ErrCode::kGood;
}

ErrCode
AlignmentFilter::flushImpl()
{
    LOG_INFO("lidar alignment datafilter is flushing");

    std::unique_lock<ImplMutex> locker(mutex());
    retentionDataMap.reset();
    return ErrCode::kGood;
}

ErrCode
AlignmentFilter::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("lidar alignment datafilter is starting");
    std::unique_lock<ImplMutex> locker(mutex());
    ErrCode code = config::CatchYamlCall([&, this]() { return parseConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse datafilter alignment config: %s failed", path.c_str());

    DS3D_FAILED_RETURN(
        (_config.lidarInputKey.size() == _config.lidarAlignedKey.size()), ErrCode::kConfig,
        "arrays lidar_input_keys and lidar_aligned_keys should use the same order of sensor data key names and have "
        "the same size\n"
        "example:\n"
        "lidar_input_keys:   [DS3D::LidarXYZI+0,           DS3D::LidarXYZI+1]\n"
        "lidar_aligned_keys: [DS3D::LidarAlignedXYZIKey+0, DS3D::LidarAlignedXYZIKey+1]");

    uint32_t pointsBytes = _config.maxPoints * (_config.lidarElemSize * sizeof(float));
    DS3D_ERROR_RETURN(
        reserveMem(_alignedOpPool, pointsBytes, _config.memPoolSize, "pointsMem"),
        "depth2point filter reserve points memory failed");

    setInputCaps(_config.compConfig.gstInCaps);
    setOutputCaps(_config.compConfig.gstOutCaps);

    retentionDataMap.reset(NvDs3d_CreateDataHashMap());

    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::filter

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataFilter*
createLidarAlignmentFilter()
{
    return NewAbiRef<abiDataFilter>(new impl::filter::AlignmentFilter);
}

DS3D_EXTERN_C_END
