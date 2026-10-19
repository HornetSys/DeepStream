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

#include "realsense_dataloader.h"
#include "realsense_external_frame.h"
#include "realsense_config.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wpessimizing-move"
#include <librealsense2/rs.hpp>
#pragma GCC diagnostic pop

namespace ds3d { namespace impl { namespace realsense {

constexpr uint64_t kMsToNas = 1000000ULL;

namespace {

bool
find_stream_serial(rs2_stream streamType, std::string& serial)
{
    rs2::context ctx;
    auto devs = ctx.query_devices();
    for (auto dev : devs) {
        for (auto& sensor : dev.query_sensors()) {
            for (auto& profile : sensor.get_stream_profiles()) {
                if (profile.stream_type() == streamType) {
                    if (dev.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)) {
                        serial = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
                    }
                    return true;
                }
            }
        }
    }
    return false;
}

ErrCode
CatchRSError(std::function<ErrCode()> f)
{
    ErrCode code = ErrCode::kGood;
    DS3D_TRY { code = f(); }
    DS3D_CATCH_ERROR(rs2::error, ErrCode::kRealSense, "rs2 error")
    DS3D_CATCH_ANY(ErrCode::kRealSense, "rs2 error")
    return code;
}

}  // namespace


struct StreamInfo {
    std::string name;
    rs2_stream type = RS2_STREAM_ANY;
    rs2_format format = RS2_FORMAT_ANY;
    std::string serial;
    uint32_t framerate = 0;
};

class RealsenseImplDataLoader : public SyncImplDataLoader {
public:
    RealsenseImplDataLoader() = default;
    ~RealsenseImplDataLoader() override = default;

protected:
    ErrCode startImpl(const std::string& content, const std::string& path) override;
    ErrCode readDataImpl(GuardDataMap& datamap) override;
    ErrCode stopImpl() override;

private:
    Config _config;
    std::shared_ptr<rs2::pipeline> _pipe;
    std::vector<StreamInfo> _streamsInfo;
    IntrinsicsParam _depthIntrinsics;
    IntrinsicsParam _colorIntrinsics;
    ExtrinsicsParam _d2cExtrinsics;
};

ErrCode
RealsenseImplDataLoader::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("Realsense dataloader is starting");
    ErrCode code =
        config::CatchYamlCall([&, this]() { return parseConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse config: %s failed", path.c_str());
    DS3D_FAILED_RETURN(_config.check(), ErrCode::kConfig, "check config: %s failed", path.c_str());

    std::vector<StreamInfo> streams;
    if (_config.hasDepthStream() || _config.hasPointsStream()) {
        streams.emplace_back(
            StreamInfo{kDepthStream, RS2_STREAM_DEPTH, RS2_FORMAT_Z16, "", _config.framerate});
    }
    if (_config.hasColorStream() || _config.hasPointsStream()) {
        streams.emplace_back(
            StreamInfo{kColorStream, RS2_STREAM_COLOR, RS2_FORMAT_RGBA8, "", _config.framerate});
    }
    DS_ASSERT(!streams.empty());

    rs2::config rsCfg;
    for (auto& stream : streams) {
        std::string serial;
        DS3D_FAILED_RETURN(
            find_stream_serial(stream.type, serial), ErrCode::kUnsupported,
            "stream: %s is not supported from device list", stream.name.c_str());
        stream.serial = serial;
        rsCfg.enable_stream(stream.type, stream.format, stream.framerate);
        LOG_INFO(
            "realsense enabled stream: %s, with serials: %s", stream.name.c_str(), serial.c_str());
    }
    _streamsInfo = std::move(streams);

    code = CatchRSError([this, &rsCfg]() {
        _pipe.reset(new rs2::pipeline());
        DS_ASSERT(_pipe);
        auto profile = _pipe->start(rsCfg);
        DS3D_FAILED_RETURN(profile, ErrCode::kRealSense, "realsense pipe start failed.");

        rs2::stream_profile depthStream;
        rs2::stream_profile colorStream;
        if (_config.hasDepthStream()) {
            depthStream = profile.get_stream(RS2_STREAM_DEPTH);
            auto stream = depthStream.as<rs2::video_stream_profile>();
            DS3D_FAILED_RETURN(
                stream, ErrCode::kRealSense, "realsense does not have a depth stream.");
            auto intr = stream.get_intrinsics();
            _depthIntrinsics.width = intr.width;
            _depthIntrinsics.height = intr.height;
            _depthIntrinsics.centerX = intr.ppx;
            _depthIntrinsics.centerY = intr.ppy;
            _depthIntrinsics.fx = intr.fx;
            _depthIntrinsics.fy = intr.fy;
        }
        if (_config.hasColorStream()) {
            colorStream = profile.get_stream(RS2_STREAM_COLOR);
            auto stream = colorStream.as<rs2::video_stream_profile>();
            DS3D_FAILED_RETURN(
                stream, ErrCode::kRealSense, "realsense does not have a color stream.");
            auto intr = stream.get_intrinsics();
            _colorIntrinsics.width = intr.width;
            _colorIntrinsics.height = intr.height;
            _colorIntrinsics.centerX = intr.ppx;
            _colorIntrinsics.centerY = intr.ppy;
            _colorIntrinsics.fx = intr.fx;
            _colorIntrinsics.fy = intr.fy;
        }

        if (_config.hasDepthStream() && _config.hasColorStream() && !_config.alignImage2Depth) {
            DS_ASSERT(depthStream);
            DS_ASSERT(colorStream);
            auto extrinsics = depthStream.get_extrinsics_to(colorStream);
            memcpy(&_d2cExtrinsics.rotation, extrinsics.rotation, sizeof(extrinsics.rotation));
            memcpy(
                &_d2cExtrinsics.translation, extrinsics.translation,
                sizeof(extrinsics.translation));
        }

        return ErrCode::kGood;
    });
    DS3D_ERROR_RETURN(code, "start realsense pipeline failed.");

    setOutputCaps(_config.compConfig.gstOutCaps);
    LOG_DEBUG("Realsense dataloader is started");

    return ErrCode::kGood;
}

namespace {
ErrCode
addPointCloud(GuardDataMap& datamap, rs2::depth_frame depth, rs2::video_frame color)
{
    DS_ASSERT(depth);
    DS_ASSERT(color);
    rs2::pointcloud pc;
    rs2::points points;
    pc.map_to(color);
    points = pc.calculate(depth);
    DS3D_FAILED_RETURN(points, ErrCode::kRealSense, "Realsense depth to cloud conversion failed");

    const rs2::vertex* pointPtr = points.get_vertices();
    const rs2::texture_coordinate* uvPtr = points.get_texture_coordinates();
    size_t num = points.size();

    FrameGuard pointXyz =
        wrapPointXYZFrame<float>((void*)pointPtr, num, MemType::kCpu, 0, [data = points](void*) {});
    DS3D_ERROR_RETURN(datamap.setGuardData(kPointXYZ, pointXyz), "set pointxyz values failed");

    FrameGuard pointUvCoord = impl::wrapPointCoordUVFrame<float>(
        (void*)uvPtr, num, MemType::kCpu, 0, [data = points](void*) {});
    DS3D_ERROR_RETURN(
        datamap.setGuardData(kPointCoordUV, pointUvCoord), "set pointUVCoord values failed");
    return ErrCode::kGood;
}
}  // namespace

ErrCode
RealsenseImplDataLoader::readDataImpl(GuardDataMap& datamap)
{
    LOG_DEBUG("Realsense dataloader is reading data");
    ErrCode code = CatchRSError([this, &datamap]() {
        auto pipe = _pipe;
        DS3D_FAILED_RETURN(pipe, ErrCode::kRealSense, "realsense pipeline stopped");
        rs2::frameset data = pipe->wait_for_frames();
        if (_config.hasColorStream() && _config.hasDepthStream() && _config.alignImage2Depth) {
            rs2::align alignToDepth(RS2_STREAM_DEPTH);
            data = alignToDepth.process(data);
        }
        auto depthFrame = data.get_depth_frame();
        auto videoFrame = data.get_color_frame();
        if (!depthFrame && !videoFrame) {
            LOG_WARNING("No depth and color frame captured.");
            return ErrCode::kByPass;
        }
        abiRefDataMap* newData = NvDs3d_CreateDataHashMap();
        datamap.reset(newData);
        TimeStamp ts{0};
        if (depthFrame) {
            if (_config.hasDepthStream()) {
                ShrdPtr<abi2DFrame> frame = std::make_shared<RSZ16DepthFrame>(depthFrame);
                DS_ASSERT(frame);
                DS3D_ERROR_RETURN(datamap.setPtrData(kDepthFrame, frame), "set depth frame failed");
                DepthScale scale{depthFrame.get_units(), {nullptr}};
                DS3D_ERROR_RETURN(
                    datamap.setData(kDepthScaleUnit, scale), "set depth frame scale factor failed");
                DS3D_ERROR_RETURN(
                    datamap.setData(kDepthIntrinsics, _depthIntrinsics),
                    "set depth frame intrinsics failed");
            }
            ts.t0 = depthFrame.get_timestamp() * kMsToNas;
        }
        if (_config.hasColorStream()) {
            DS_ASSERT(videoFrame);
            ShrdPtr<abi2DFrame> frame = std::make_shared<RSRGBAFrame>(videoFrame);
            DS_ASSERT(frame);
            DS3D_ERROR_RETURN(datamap.setPtrData(kColorFrame, frame), "set color frame failed");

            uint64_t t = videoFrame.get_timestamp() * kMsToNas;
            if (t > ts.t0) {
                ts.t0 = t;
            }
        }

        if (_config.hasColorStream() && _config.hasDepthStream()) {
            if (_config.alignImage2Depth) {
                DS3D_ERROR_RETURN(
                    datamap.setData<bool>(kColorDepthAligned, true),
                    "set ColorDepthAligned value failed");
            } else {
                DS3D_ERROR_RETURN(
                    datamap.setData(kColorIntrinsics, _colorIntrinsics),
                    "set color frame intrinsics failed");
                DS3D_ERROR_RETURN(
                    datamap.setData(kDepth2ColorExtrinsics, _d2cExtrinsics),
                    "set depth2color extrinsics failed");
            }
        }

        if (depthFrame && videoFrame && _config.hasPointsStream()) {
            DS3D_ERROR_RETURN(
                addPointCloud(datamap, depthFrame, videoFrame), "add point-cloud data failed");
        }
        DS3D_ERROR_RETURN(datamap.setData(kTimeStamp, ts), "set timestamp failed");
        return ErrCode::kGood;
    });
    DS3D_ERROR_RETURN(code, "start realsense pipeline failed.");
    LOG_DEBUG("Realsense dataloader read data successfully.");
    return ErrCode::kGood;
}

ErrCode
RealsenseImplDataLoader::stopImpl()
{
    LOG_INFO("Realsense dataloader is closing");

    if (_pipe) {
        ErrCode c = CatchRSError([this]() {
            _pipe->stop();
            return ErrCode::kGood;
        });
        _pipe.reset();
        DS3D_ERROR_RETURN(c, "start realsense pipeline failed.");
    }
    LOG_DEBUG("Realsense dataloader is closed");
    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::realsense

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataLoader*
createRealsenseDataloader()
{
    return NewAbiRef<abiDataLoader>(new impl::realsense::RealsenseImplDataLoader);
}
DS3D_EXTERN_C_END