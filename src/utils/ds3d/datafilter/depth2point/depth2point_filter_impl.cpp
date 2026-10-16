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

#include "depth2point_filter_impl.h"
#include "ds3d/common/impl/impl_frames.h"

#include <unistd.h>

namespace ds3d { namespace impl { namespace filter {

class InProcess {
    volatile bool& _inFlag;

public:
    InProcess(volatile bool& flag) : _inFlag(flag) { _inFlag = flag; }
    ~InProcess() { release(); }
    void release() { _inFlag = false; }
};

namespace {
ErrCode
reserveMem(
    std::shared_ptr<BufferPool<MemPtr>>& pool, size_t memSize, uint32_t count,
    const std::string& name)
{
    pool = std::make_shared<BufferPool<MemPtr>>(name);
    DS_ASSERT(pool);
    for (uint32_t i = 0; i < count; ++i) {
        MemPtr ptr = CpuMemBuf::CreateBuf(memSize);
        pool->setBuffer(std::move(ptr));
    }
    DS3D_FAILED_RETURN(
        pool->size() == count, ErrCode::kMem, "allocate CPU memory failed on %s", name.c_str());
    return ErrCode::kGood;
}

template <typename DepthT>
void
alignedDepth2PointCalculate(
    DepthT* depthPtr, float depthScale, const IntrinsicsParam& depthIntrin, vec3f* point, vec2f* uv,
    uint32_t width, uint32_t height)
{
    for (uint32_t y = 0; y < height; ++y) {
        float v = (float)y / (float)height;
        for (uint32_t x = 0; x < width; ++x) {
            uv->x = (float)x / (float)width;
            uv->y = v;
            float d = (float)(*depthPtr) * depthScale;
            point->z = d;
            point->x = ((float)x - depthIntrin.centerX) * d / depthIntrin.fx;
            point->y = ((float)y - depthIntrin.centerY) * d / depthIntrin.fy;
            ++point;
            ++uv;
            ++depthPtr;
        }
    }
}

inline vec3f
transform(const vec3f& point, const ExtrinsicsParam& ext)
{
    vec3f ret;
    ret.x = ext.rotation[0].x * point.x + ext.rotation[1].x * point.y +
            ext.rotation[2].x * point.z + ext.translation.x;
    ret.y = ext.rotation[0].y * point.x + ext.rotation[1].y * point.y +
            ext.rotation[2].y * point.z + ext.translation.y;
    ret.z = ext.rotation[0].z * point.x + ext.rotation[1].z * point.y +
            ext.rotation[2].z * point.z + ext.translation.z;
    return ret;
}

template <typename DepthT>
void
unalignedDepth2PointCalculate(
    DepthT* depthPtr, float depthScale, const IntrinsicsParam& depthIntrin,
    const ExtrinsicsParam& d2cExtrin, const IntrinsicsParam& colorIntrin, vec3f* point, vec2f* uv,
    uint32_t width, uint32_t height)
{
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            float d = (float)(*depthPtr) * depthScale;
            point->z = d;
            point->x = ((float)x - depthIntrin.centerX) * d / depthIntrin.fx;
            point->y = ((float)y - depthIntrin.centerY) * d / depthIntrin.fy;
            vec3f tfPoint = transform(*point, d2cExtrin);
            vec2f tfUV{
                {(tfPoint.x / tfPoint.z) * colorIntrin.fx + colorIntrin.centerX,
                 (tfPoint.y / tfPoint.z) * colorIntrin.fy + colorIntrin.centerY}};
            uv->x = tfUV.x / (float)colorIntrin.width;
            uv->y = tfUV.y / (float)colorIntrin.height;
            ++point;
            ++uv;
            ++depthPtr;
        }
    }
}

}  // namespace

Depth2PointFilter::~Depth2PointFilter()
{
    stop_i();
}

ErrCode
Depth2PointFilter::processImpl(
    GuardDataMap datamap, OnGuardDataCBImpl outputDataCb, OnGuardDataCBImpl inputConsumedCb)
{
    LOG_DEBUG("datafilter is starting to filter datamap");
    std::unique_lock<ImplMutex> locker(mutex());
    InProcess inProcess(_inProcess);
    Frame2DGuard colorFrame;
    Frame2DGuard depthFrame;

    // check frames with config file
    if (_config.hasDepthStream()) {
        DS3D_FAILED_RETURN(
            datamap.hasData(kDepthFrame), ErrCode::kConfig,
            "depth2points configured depth stream as input but no depth frame found in datamap");
    }
    if (_config.hasColorStream()) {
        DS3D_FAILED_RETURN(
            datamap.hasData(kColorFrame), ErrCode::kConfig,
            "depth2points configured color stream as input but no color frame found in datamap");
    }

    // read all data out from datamap
    if (_config.hasColorStream()) {
        DS3D_ERROR_RETURN(
            datamap.getGuardData(kColorFrame, colorFrame),
            "No color data found in datamap in depth2point filter");
        if (datamap.hasData(kColorIntrinsics)) {
            IntrinsicsParam intrinsics;
            DS3D_ERROR_RETURN(
                datamap.getData(kColorIntrinsics, intrinsics), "get color intrinsics failed.");
            _colorIntrinsics = intrinsics;
        }
    }

    if (_config.hasDepthStream()) {
        DS3D_ERROR_RETURN(
            datamap.getGuardData(kDepthFrame, depthFrame),
            "No image found in datamap in depth2point filter");
        if (datamap.hasData(kDepthScaleUnit)) {
            DepthScale scale;
            DS3D_ERROR_RETURN(datamap.getData(kDepthScaleUnit, scale), "get depth scale failed.");
            _depthScale = scale.scaleUnit;
        }
        if (datamap.hasData(kDepthIntrinsics)) {
            IntrinsicsParam intrinsics;
            DS3D_ERROR_RETURN(
                datamap.getData(kDepthIntrinsics, intrinsics), "get depth intrinsics failed.");
            _depthIntrinsics = intrinsics;
        }
    }
    bool colorDepthAligned = false;
    if (datamap.hasData(kColorDepthAligned)) {
        DS3D_ERROR_RETURN(
            datamap.getData(kColorDepthAligned, colorDepthAligned),
            "get ColorDepthAligned value failed.");
    } else if (datamap.hasData(kDepth2ColorExtrinsics)) {
        ExtrinsicsParam extrinsics;
        DS3D_ERROR_RETURN(
            datamap.getData(kDepth2ColorExtrinsics, extrinsics),
            "get Depth2ColorExtrinsics value failed.");
        _d2cExtrinsics = extrinsics;
    } else {
        colorDepthAligned = true;
    }

    DS3D_FAILED_RETURN(
        depthFrame, ErrCode::kConfig,
        "depth2points process failed, no depth frame found in datamap");
    const auto& plane = depthFrame->getPlane(0);
    uint32_t points = plane.width * plane.height;
    DS3D_FAILED_RETURN(
        points <= _config.maxPoints, ErrCode::kConfig,
        "depth frame width: %u x height: %u is greater than max_points: %u", plane.width,
        plane.height, _config.maxPoints);

    FrameGuard pointXyzFrame;
    FrameGuard pointUvCoord;
    DS3D_ERROR_RETURN(
        processDepthToPoint(depthFrame, colorDepthAligned, pointXyzFrame, pointUvCoord),
        "failed to process depth to points");
    // notify the application(or gst plugins) the input data is consumed.
    locker.unlock();
    inputConsumedCb(ErrCode::kGood, datamap);
    locker.lock();

    /**
     * User can use the same datamap to keep all values or creat a new datamap if expected all data
     * to be new
     */
    DS3D_ERROR_RETURN(datamap.setGuardData(kPointXYZ, pointXyzFrame), "set point xyz frame failed");
    DS3D_ERROR_RETURN(
        datamap.setGuardData(kPointCoordUV, pointUvCoord), "set point uv coordinates failed");
    inProcess.release();
    locker.unlock();
    outputDataCb(ErrCode::kGood, datamap);

    return ErrCode::kGood;
}

ErrCode
Depth2PointFilter::processDepthToPoint(
    Frame2DGuard& depthFrame, bool colorDepthAligned, FrameGuard& pointXyzFrame,
    FrameGuard& pointUvCoord)
{
    DS_ASSERT(depthFrame);
    const auto& plane = depthFrame->getPlane(0);
    uint32_t pointNum = plane.width * plane.height;

    DS3D_FAILED_RETURN(
        depthFrame->frameType() == FrameType::kDepth && depthFrame->dataType() == DataType::kUint16,
        ErrCode::kUnsupported, "depth2points only support uint16 datatype");

    DS_ASSERT(_pointMem);
    DS_ASSERT(_uvMem);
    std::shared_ptr<MemData> points = _pointMem->acquireBuffer();
    std::shared_ptr<MemData> uvData = _uvMem->acquireBuffer();
    vec3f* pointPtr = (vec3f*)(points->data);
    vec2f* uvPtr = (vec2f*)(uvData->data);

    DS_ASSERT(points->byteSize >= pointNum * sizeof(vec3f));
    DS_ASSERT(uvData->byteSize >= pointNum * sizeof(vec2f));

    uint16_t* depthPtr = (uint16_t*)depthFrame->base();
    if (colorDepthAligned) {
        alignedDepth2PointCalculate<uint16_t>(
            depthPtr, _depthScale, _depthIntrinsics, pointPtr, uvPtr, plane.width, plane.height);
    } else {
        unalignedDepth2PointCalculate<uint16_t>(
            depthPtr, _depthScale, _depthIntrinsics, _d2cExtrinsics, _colorIntrinsics, pointPtr,
            uvPtr, plane.width, plane.height);
    }

    pointXyzFrame = wrapPointXYZFrame<float>(
        (void*)pointPtr, pointNum, MemType::kCpu, 0, [points = std::move(points)](void*) {});

    pointUvCoord = wrapPointCoordUVFrame<float>(
        (void*)uvPtr, pointNum, MemType::kCpu, 0, [uvData = std::move(uvData)](void*) {});
    return ErrCode::kGood;
}

void
Depth2PointFilter::timeoutWaitProcess()
{
    int times = 0;
    while (_inProcess && times++ < 5) {
        usleep(1000 * 1000);
    }
}

ErrCode
Depth2PointFilter::stopImpl()
{
    LOG_INFO("datafilter is closing");
    std::unique_lock<ImplMutex> locker(mutex());
    timeoutWaitProcess();
    _pointMem.reset();
    _uvMem.reset();

    return ErrCode::kGood;
}

ErrCode
Depth2PointFilter::flushImpl()
{
    LOG_INFO("datafilter is flushing");

    std::unique_lock<ImplMutex> locker(mutex());
    timeoutWaitProcess();
    return ErrCode::kGood;
}

ErrCode
Depth2PointFilter::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("depthcolor datafilter is starting");
    std::unique_lock<ImplMutex> locker(mutex());
    ErrCode code =
        config::CatchYamlCall([&, this]() { return parseConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse datafilter depth2point config: %s failed", path.c_str());
    DS3D_FAILED_RETURN(
        _config.check(), ErrCode::kConfig, "check datafilter depth2point config: %s failed",
        path.c_str());

    uint32_t pointsBytes = _config.maxPoints * sizeof(vec3f);
    uint32_t uvBytes = _config.maxPoints * sizeof(vec2f);
    DS3D_ERROR_RETURN(
        reserveMem(_pointMem, pointsBytes, _config.memPoolSize, "pointsMem"),
        "depth2point filter reserve points memory failed");
    DS3D_ERROR_RETURN(
        reserveMem(_uvMem, uvBytes, _config.memPoolSize, "uvMem"),
        "depth2point filter reserve uv memory failed");

    setInputCaps(_config.compConfig.gstInCaps);
    setOutputCaps(_config.compConfig.gstOutCaps);

    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::filter

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataFilter*
createDepth2PointFilter()
{
    return NewAbiRef<abiDataFilter>(new impl::filter::Depth2PointFilter);
}

DS3D_EXTERN_C_END