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

#include "depth_color_source.h"
#include "depth_source_config.h"
#include "ds3d/common/hpp/profiling.hpp"
#include "ds3d/common/impl/impl_frames.h"

namespace ds3d { namespace impl { namespace depthsource {

constexpr uint64_t kMsToNas = 1000000ULL;

class DepthColorFileSourceImpl : public SyncImplDataLoader {
public:
    DepthColorFileSourceImpl() = default;
    ~DepthColorFileSourceImpl() override = default;

protected:
    ErrCode startImpl(const std::string& content, const std::string& path) override;
    ErrCode readDataImpl(GuardDataMap& datamap) override;
    ErrCode stopImpl() override;

private:
    Config _config;
    profiling::FileReader _depthReader;
    profiling::FileReader _colorReader;
};

ErrCode
DepthColorFileSourceImpl::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("DepthColorSource dataloader is starting");
    ErrCode code =
        config::CatchYamlCall([&, this]() { return parseConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse config: %s failed", path.c_str());

    if (_config.hasDepthSource()) {
        DS3D_FAILED_RETURN(
            _depthReader.open(_config.depthSource), ErrCode::kConfig,
            "open depth source file: %s failed.", _config.depthSource.c_str());
    }

    if (_config.hasColorSource()) {
        DS3D_FAILED_RETURN(
            _colorReader.open(_config.colorSource), ErrCode::kConfig,
            "open color source file: %s failed.", _config.colorSource.c_str());
    }

    setOutputCaps(_config.compConfig.gstOutCaps);
    LOG_DEBUG("DepthColorSource dataloader is started");

    return ErrCode::kGood;
}

ErrCode
DepthColorFileSourceImpl::readDataImpl(GuardDataMap& outData)
{
    LOG_DEBUG("DepthColorSource dataloader is reading data");

    GuardDataMap datamap(NvDs3d_CreateDataHashMap(), true);

    if (_config.hasDepthSource()) {
        uint32_t bytesPerFrame = _config.depthPlane.pitchInBytes * _config.depthPlane.height;
        DS_ASSERT(bytesPerFrame);
        std::vector<uint8_t> data(bytesPerFrame);
        if (_depthReader.read(&data[0], bytesPerFrame) != (int32_t)bytesPerFrame) {
            LOG_INFO("Depth source read data failed. return EOS");
            return ErrCode::KEndOfStream;
        }
        void* dataPtr = &data[0];
        Frame2DGuard frame = Wrap2DFrame<uint16_t, FrameType::kDepth>(
            dataPtr, {_config.depthPlane}, bytesPerFrame, MemType::kCpu, 0,
            [data = std::move(data)](void*) {});
        DS3D_FAILED_RETURN(frame, ErrCode::kMem, "generate depth 2d frame failed.");
        DS3D_ERROR_RETURN(
            datamap.setGuardData(kDepthFrame, frame), "datamap set depth frame failed.");
        DepthScale scale{_config.depthScale, {nullptr}};
        DS3D_ERROR_RETURN(
            datamap.setData(kDepthScaleUnit, scale), "datamap set depth scale failed");

        if (_config.depthIntrinIsSet) {
            DS3D_ERROR_RETURN(
                datamap.setData(kDepthIntrinsics, _config.depthIntrinsics),
                "set depth frame intrinsics failed");
        }
    }
    if (_config.hasColorSource()) {
        uint32_t bytesPerFrame = _config.colorPlane.pitchInBytes * _config.colorPlane.height;
        DS_ASSERT(bytesPerFrame);
        std::vector<uint8_t> data(bytesPerFrame);
        if (_colorReader.read(&data[0], bytesPerFrame) != (int32_t)bytesPerFrame) {
            LOG_INFO("Depth source read data failed. return EOS");
            return ErrCode::KEndOfStream;
        }
        void* dataPtr = &data[0];
        Frame2DGuard frame = Wrap2DFrame<uint8_t, FrameType::kColorRGBA>(
            dataPtr, {_config.colorPlane}, bytesPerFrame, MemType::kCpu, 0,
            [data = std::move(data)](void*) {});
        DS3D_FAILED_RETURN(frame, ErrCode::kMem, "generate color 2d frame failed.");
        DS3D_ERROR_RETURN(
            datamap.setGuardData(kColorFrame, frame), "datamap set color frame failed.");
        if (_config.colorIntrinIsSet) {
            DS3D_ERROR_RETURN(
                datamap.setData(kColorIntrinsics, _config.colorIntrinsics),
                "set color frame intrinsics failed");
        }
    }

    if (_config.hasColorSource() && _config.hasDepthSource() && _config.depth2ColorExSet) {
        DS3D_ERROR_RETURN(
            datamap.setData(kDepth2ColorExtrinsics, _config.d2cExtrinsics),
            "set depth2color extrinsics failed");
    }

    outData = std::move(datamap);
    LOG_DEBUG("DepthColorSource dataloader read data successfully.");
    return ErrCode::kGood;
}

ErrCode
DepthColorFileSourceImpl::stopImpl()
{
    LOG_INFO("DepthColorSource dataloader is closing");

    _depthReader.close();
    _colorReader.close();
    LOG_DEBUG("DepthColorSource dataloader is closed");
    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::depthsource

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataLoader*
createDepthColorLoader()
{
    return NewAbiRef<abiDataLoader>(new impl::depthsource::DepthColorFileSourceImpl);
}
DS3D_EXTERN_C_END