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

#include "sensorfusion_cluster_filter_impl.h"
#include "ds3d/common/impl/impl_frames.h"

#include "ds3d/common/ds3d_analysis_datatype.h"

namespace ds3d { namespace impl { namespace filter {

SensorFusionClusterFilter::~SensorFusionClusterFilter()
{
    stop_i();
}

ErrCode
SensorFusionClusterFilter::getCalibrationData(GuardDataMap& datamap)
{
    std::string intKey = _config.camIntrinsicsKey;
    std::string intParamKey = _config.camIntrinsicsParamKey;
    std::string extKey = _config.lidar2CamExtrinsicsKey;

    // camera intrinsic param settings are required
    DS3D_FAILED_RETURN(
        datamap.hasData(intParamKey), ErrCode::kParam,
        "Cam intrinsic param key: %s is not found in datamap during sensor fusion cluster",
        intParamKey.c_str());
    IntrinsicsParam param;
    DS3D_ERROR_RETURN(
        datamap.getData(intParamKey, param),
        "Cam intrinsic param: %s type error in datamap during sensor fusion cluster",
        intParamKey.c_str());
    _camIntrinsicParam = param;

    if (datamap.hasData(intKey)) {
        TransformMatrix intMat;
        DS3D_ERROR_RETURN(
            datamap.getData(intKey, intMat),
            "Cam Intrinsic: %s type error in datamap during sensor fusion cluster", intKey.c_str());
        _camIntrinsicMat = intMat;
    }

    if (datamap.hasData(extKey)) {
        TransformMatrix extMat;
        DS3D_ERROR_RETURN(
            datamap.getData(extKey, extMat),
            "Lidar to Cam Extrinsic: %s type error in datamap during sensor fusion cluster",
            extKey.c_str());
        _lidar2camMat = extMat;
    }

    if (!datamap.hasData(intKey)) {
        _camIntrinsicMat = getIntrinsicMat(_camIntrinsicParam);
    }

    // get color image size
    std::string imageKey = _config.input2DImageKey;
    if (datamap.hasData(imageKey)) {
        Frame2DGuard imageFrame;
        DS3D_ERROR_RETURN(
            datamap.getGuardData(imageKey, imageFrame),
            "Color image: %s type error in datamap during fusion filter", imageKey.c_str());
        DS_ASSERT(imageFrame);
        Frame2DPlane plane = imageFrame->getPlane(0);
        _imageWidth = plane.width;
        _imageHeight = plane.height;
    } else {
        _imageWidth = _camIntrinsicParam.width;
        _imageHeight = _camIntrinsicParam.height;
    }

    LOG_DEBUG(
        "Fusion: get calibration data for intrinsic(w:%d, h:%d, cx:%4.3f, cy:%4.3f)",
        _camIntrinsicParam.width, _camIntrinsicParam.height, _camIntrinsicParam.centerX,
        _camIntrinsicParam.centerY);
    LOG_DEBUG("Fusion: get color image size(w:%d, h:%d)", _imageWidth, _imageHeight);

    DS3D_FAILED_RETURN(
        _camIntrinsicParam.width > 0 && _camIntrinsicParam.height > 0 && _imageHeight > 0 &&
            _imageWidth > 0,
        ErrCode::kParam, "Fusion: calibration or image size should not be 0");

    _transform = gl::mat4Mul(
        gl::mat4f(std::begin(_camIntrinsicMat.matrix), std::end(_camIntrinsicMat.matrix)),
        gl::mat4f(std::begin(_lidar2camMat.matrix), std::end(_lidar2camMat.matrix)));
    gl::mat4f scale = {
        {{(float)_imageWidth / (float)_camIntrinsicParam.width, 0, 0, 0}},
        {{0, (float)_imageHeight / (float)_camIntrinsicParam.height, 0, 0}},
        {{0, 0, 1, 0}},
        {{0, 0, 0, 1}},
    };
    _transform = gl::mat4Mul(scale, _transform);
    DS_ASSERT(_transform.size() == 4);

    return ErrCode::kGood;
}

namespace {

void
Print3D2DBbox(Lidar3DBbox* obj3D, uint32_t num3DBbox, Object2DBbox* obj2D, uint32_t num2DBbox)
{
    LOG_DEBUG("Fusion Received %d 3D bbox.", num3DBbox);
    for (uint32_t i = 0; i < num3DBbox; ++i) {
        LOG_DEBUG(
            "[%5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f], score: %2.3f, %s", obj3D[i].centerX,
            obj3D[i].centerY, obj3D[i].centerZ, obj3D[i].dx, obj3D[i].dy, obj3D[i].dz,
            obj3D[i].score, obj3D[i].labels);
    }
    LOG_DEBUG("Fusion Received %d 2D bbox.", num2DBbox);
    for (uint32_t i = 0; i < num2DBbox; ++i) {
        LOG_DEBUG(
            "[%5.3f, %5.3f, %5.3f, %5.3f], score: %2.3f, %s", obj2D[i].centerX, obj2D[i].centerY,
            obj2D[i].dx, obj2D[i].dy, obj2D[i].score, obj2D[i].labels);
    }
}

void
project3D(const Lidar3DBbox& obj3D, const glm::mat4& t, vec2f& tl, vec2f& br)
{
    auto projectFunc = [&t](const vec3f& p) {
        auto out = t * glm::vec4(p.x, p.y, p.z, 1.0f);
        return vec2f{{out.x / out.z, out.y / out.z}};
    };
    static const std::vector<vec3f> diretion = {
        // negtive y plane
        {{-0.5f, -0.5f, -0.5f}},
        {{0.5f, -0.5f, -0.5f}},
        {{0.5f, -0.5f, 0.5f}},
        {{-0.5f, -0.5f, 0.5f}},
        // positive y plane
        {{-0.5f, 0.5f, -0.5f}},
        {{0.5f, 0.5f, -0.5f}},
        {{0.5f, 0.5f, 0.5f}},
        {{-0.5f, 0.5f, 0.5f}},
    };
    float cosYaw = cos(obj3D.yaw);
    float sinYaw = sin(obj3D.yaw);

    // calculate top-left and bottom-right of 3D points projected into 2D area
    for (uint32_t iD = 0; iD < diretion.size(); ++iD) {
        vec3f p{
            {obj3D.centerX + diretion[iD].x * obj3D.dx * cosYaw +
                 diretion[iD].y * obj3D.dy * (-sinYaw),
             obj3D.centerY + diretion[iD].x * obj3D.dx * sinYaw +
                 diretion[iD].y * obj3D.dy * cosYaw,
             obj3D.centerZ + diretion[iD].z * obj3D.dz}};
        vec2f proj = projectFunc(p);
        if (iD == 0) {
            tl = proj;
            br = proj;
        } else {
            tl.x = std::min(tl.x, proj.x);
            tl.y = std::min(tl.y, proj.y);
            br.x = std::max(br.x, proj.x);
            br.y = std::max(br.y, proj.y);
        }
    }
}

float
iouMatch(const vec2f& tl, const vec2f& br, const Object2DBbox& obj2D)
{
    vec2f tl2D{{obj2D.centerX - obj2D.dx / 2.0f, obj2D.centerY - obj2D.dy / 2.0f}};
    vec2f br2D{{obj2D.centerX + obj2D.dx / 2.0f, obj2D.centerY + obj2D.dy / 2.0f}};

    // calcualate overlap area
    vec2f overlap0{{std::max(tl.x, tl2D.x), std::max(tl.y, tl2D.y)}};
    vec2f overlap1{{std::min(br.x, br2D.x), std::min(br.y, br2D.y)}};
    if (overlap1.x <= overlap0.x || overlap1.y <= overlap0.y) {
        return -1.0f;
    }

    float areaIou = (overlap1.x - overlap0.x) * (overlap1.y - overlap0.y);
    float areaProj = (br.x - tl.x) * (br.y - tl.y);
    float iouValue = areaIou / (areaProj + obj2D.dx * obj2D.dy - areaIou);

    return iouValue;
}

}  // namespace

ErrCode
SensorFusionClusterFilter::FuseDetectionCluster(
    Lidar3DBbox* obj3DBase, uint32_t num3DBbox, Object2DBbox* obj2DBase, uint32_t num2DBbox,
    std::vector<FusedDetection>& fusedVec)
{
    glm::mat4 t = gl::dsMat4ToGlm(_transform);

    for (uint32_t i3D = 0; i3D < num3DBbox; ++i3D) {
        const Lidar3DBbox& obj3D = obj3DBase[i3D];
        auto iteHash = _hashMatches.find(obj3D.labels);
        if (iteHash == _hashMatches.end()) {
            LOG_DEBUG("3d object: %s is skipped.", obj3D.labels);
            continue;
        }
        const auto& matchList = iteHash->second;

        vec2f tl{{0.0f}}, br{{0.0f}};
        project3D(obj3D, t, tl, br);
        LOG_DEBUG(
            "3D object: %s, [%5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f] projected into "
            "2D location [%5.3f, %5.3f, %5.3f, %5.3f], looking for iou matches",
            obj3D.labels, obj3D.centerX, obj3D.centerY, obj3D.centerZ, obj3D.dx, obj3D.dy, obj3D.dz,
            tl.x, tl.y, br.x - tl.x, br.y - tl.y);

        FusedDetection best;
        float bestScore = -1.0f;
        best.obj3D = obj3D;

        // search 2D bbox to get the best match
        for (uint32_t i2D = 0; i2D < num2DBbox; ++i2D) {
            const Object2DBbox& obj2D = obj2DBase[i2D];
            auto iMatch = std::find_if(
                matchList.cbegin(), matchList.cend(),
                [&obj2D](const MatchLabel& m) { return m.label2D == obj2D.labels; });
            if (iMatch == matchList.cend()) {
                continue;
            }
            MatchLabel match = *iMatch;
            float matchScore = iouMatch(tl, br, obj2D);
            if (matchScore < 0) {
                continue;
            }
            if (matchScore < _config.iouThreshold || matchScore <= bestScore) {
                continue;
            }
            LOG_DEBUG(
                "best 2D match: [%5.3f, %5.3f, %5.3f, %5.3f], match score: %1.3f", obj2D.centerX,
                obj2D.centerY, obj2D.dx, obj2D.dy, matchScore);

            bestScore = matchScore;
            best.obj2D = obj2D;
            snprintf(best.labels, sizeof(best.labels), "%s", match.newLabel.c_str());
            best.bboxColor = match.color;
            best.score = matchScore;
        }

        // copy the reset of 3D bbox into fusion detection
        if (bestScore <= 0.0f) {
            best.bboxColor = obj3D.bboxColor;
            snprintf(best.labels, sizeof(best.labels), "%s", obj3D.labels);
            best.score = 0.0f;
        }
        fusedVec.emplace_back(best);
    }
    return ErrCode::kGood;
}

ErrCode
SensorFusionClusterFilter::processImpl(
    GuardDataMap datamap, OnGuardDataCBImpl outputDataCb, OnGuardDataCBImpl inputConsumedCb)
{
    LOG_DEBUG("sensorfusion is fusing 3D and 2D objects, datamap keys: ");
    datamap.printDebug();

    std::unique_lock<ImplMutex> locker(mutex());
    std::string input2DBboxkey = _config.input2DBboxKey;
    std::string input3Dbboxkey = _config.input3DbboxKey;

    if (!datamap.hasData(input2DBboxkey) || !datamap.hasData(input3Dbboxkey)) {
        LOG_DEBUG("2D Bbox and 3D Bbox are not co-exist, skip this frame");
        return ErrCode::kByPass;
    }

    // extract 3d lidar and 2d video bbox from datamap
    FrameGuard lidar3dBboxData;
    FrameGuard video2dBboxData;
    DS3D_ERROR_RETURN(
        datamap.getGuardData(input3Dbboxkey, lidar3dBboxData),
        "No 3d bbox data: %s found in datamap in sensor fusion", input3Dbboxkey.c_str());
    DS3D_ERROR_RETURN(
        datamap.getGuardData(input2DBboxkey, video2dBboxData),
        "No 2d bbox data: %s found in datamap in sensor fusion", input2DBboxkey.c_str());

    DS3D_ERROR_RETURN(
        getCalibrationData(datamap), "Calibration data is not correct in sensor fusion");

    Shape obj3DShape = lidar3dBboxData->shape();  // 1 x N x sizeof(Lidar3DBbox)
    DS_ASSERT(obj3DShape.numDims);
    Shape obj2DShape = video2dBboxData->shape();  // N x sizeof(Lidar3DBbox)
    DS_ASSERT(obj2DShape.numDims);
    if (!ShapeSize(obj3DShape) || !ShapeSize(obj2DShape)) {
        LOG_DEBUG("lidar 3D bbox or video 2D bbox number is 0");
        return ErrCode::kByPass;
    }

    uint32_t per3DObjBytes = obj3DShape.d[obj3DShape.numDims - 1];
    uint32_t per2DObjBytes = obj2DShape.d[obj2DShape.numDims - 1];
    DS3D_FAILED_RETURN(
        per3DObjBytes == sizeof(Lidar3DBbox), ErrCode::kIncompatible,
        "per 3DBbox bytes(: %d) is not compatible with sizeof Lidar3DBbox(: %d), please check "
        "version",
        (int)per3DObjBytes, (int)sizeof(Lidar3DBbox));
    DS3D_FAILED_RETURN(
        per2DObjBytes == sizeof(Object2DBbox), ErrCode::kIncompatible,
        "per 2DBbox bytes(: %d) is not compatible with sizeof Object2DBbox(: %d), please check "
        "version",
        (int)per2DObjBytes, (int)sizeof(Object2DBbox));
    uint32_t num3DBbox = ShapeSize(obj3DShape) / per3DObjBytes;
    uint32_t num2DBbox = ShapeSize(obj2DShape) / per2DObjBytes;

    Lidar3DBbox* obj3DBase = (Lidar3DBbox*)lidar3dBboxData->base();
    Object2DBbox* obj2DBase = (Object2DBbox*)video2dBboxData->base();

    if (ENABLE_DEBUG) {
        Print3D2DBbox(obj3DBase, num3DBbox, obj2DBase, num2DBbox);
    }

    // fuse through IOU cluster
    std::vector<FusedDetection> fusedResults;
    fusedResults.reserve(num3DBbox);
    DS3D_ERROR_RETURN(
        FuseDetectionCluster(obj3DBase, num3DBbox, obj2DBase, num2DBbox, fusedResults),
        "fused 3D&2D objects bbox failed");

    DS3D_ERROR_RETURN(
        attachFusedResults(datamap, fusedResults), "attach fused results into datamap failed");

    inputConsumedCb(ErrCode::kGood, datamap);
    outputDataCb(ErrCode::kGood, datamap);

    return ErrCode::kGood;
}

ErrCode
SensorFusionClusterFilter::attachFusedResults(
    GuardDataMap& datamap, std::vector<FusedDetection>& fusedVec)
{
    if (fusedVec.empty()) {
        return ErrCode::kGood;
    }

    // attach new lidar 3D bbox as fused result for rendering
    if (!_config.outputFusedBboxRenderKey.empty()) {
        std::vector<Lidar3DBbox> renderBbox(fusedVec.size());
        for (uint32_t i = 0; i < fusedVec.size(); ++i) {
            Lidar3DBbox& bbox = renderBbox[i];
            bbox = fusedVec[i].obj3D;
            bbox.bboxColor = fusedVec[i].bboxColor;
            bbox.score = fusedVec[i].score;
            snprintf(bbox.labels, sizeof(bbox.labels), "%s", fusedVec[i].labels);
        }
        void* bboxBase = &renderBbox[0];
        size_t bboxBytes = renderBbox.size() * sizeof(Lidar3DBbox);
        Shape bboxShape{2, {(int)renderBbox.size(), sizeof(Lidar3DBbox)}};
        FrameGuard renderbboxFrame = impl::WrapFrame<uint8_t, FrameType::kCustom>(
            bboxBase, bboxBytes, bboxShape, MemType::kCpu, 0,
            [keep = std::move(renderBbox)](void*) {});
        DS3D_ERROR_RETURN(
            datamap.setGuardData(_config.outputFusedBboxRenderKey, renderbboxFrame),
            "attach fused render bbox (key: %s) into datamap failed.",
            _config.outputFusedBboxRenderKey.c_str());
    }

    // attach fused detection frame into datamap
    void* fuseBase = &fusedVec[0];
    size_t fuseBytes = fusedVec.size() * sizeof(FusedDetection);
    Shape fuseShape{2, {(int)fusedVec.size(), sizeof(FusedDetection)}};
    FrameGuard fusedFrame = impl::WrapFrame<uint8_t, FrameType::kCustom>(
        fuseBase, fuseBytes, fuseShape, MemType::kCpu, 0, [keep = std::move(fusedVec)](void*) {});
    DS3D_ERROR_RETURN(
        datamap.setGuardData(_config.outputFusedKey, fusedFrame),
        "attach fused frame (key: %s) into datamap failed.", _config.outputFusedKey.c_str());
    return ErrCode::kGood;
}

ErrCode
SensorFusionClusterFilter::stopImpl()
{
    LOG_INFO("sensorfusion is closing");
    std::unique_lock<ImplMutex> locker(mutex());

    return ErrCode::kGood;
}

ErrCode
SensorFusionClusterFilter::flushImpl()
{
    LOG_INFO("sensorfusion is flushing");

    std::unique_lock<ImplMutex> locker(mutex());

    return ErrCode::kGood;
}

ErrCode
SensorFusionClusterFilter::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("sensorfusion is starting");
    std::unique_lock<ImplMutex> locker(mutex());
    ErrCode code =
        config::CatchYamlCall([&, this]() { return parseConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse datafilter lidarinference config: %s failed", path.c_str());

    for (const auto& match : _config.fusionLabels) {
        _hashMatches[match.label3D].emplace_back(match);
    }

    setInputCaps(_config.compConfig.gstInCaps);
    setOutputCaps(_config.compConfig.gstOutCaps);

    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::filter

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataFilter*
NvDS3D_CreateFilter_SensorFusionCluster()
{
    return NewAbiRef<abiDataFilter>(new impl::filter::SensorFusionClusterFilter);
}

DS3D_EXTERN_C_END
