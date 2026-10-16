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

#ifndef _DS3D_DATALOADER_DEPTHSOURCE_CONFIG_H
#define _DS3D_DATALOADER_DEPTHSOURCE_CONFIG_H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"

namespace ds3d { namespace impl { namespace depthsource {

constexpr const char* kDepthStream = "depth";
constexpr const char* kColorStream = "color";

struct Config {
    config::ComponentConfig compConfig;
    std::string depthSource;
    std::string colorSource;
    double depthScale = 1.0f;
    DataType depthDataType = DataType::kUint16;
    Frame2DPlane depthPlane = {640, 480, 0, sizeof(uint16_t), 0};

    FrameType colorType = FrameType::kColorRGBA;
    Frame2DPlane colorPlane = {1920, 1080, 0, sizeof(vec4b), 0};

    IntrinsicsParam depthIntrinsics;
    IntrinsicsParam colorIntrinsics;
    ExtrinsicsParam d2cExtrinsics;
    bool depthIntrinIsSet = false;
    bool colorIntrinIsSet = false;
    bool depth2ColorExSet = false;

    bool hasDepthSource() const { return !colorSource.empty(); }
    bool hasColorSource() const { return !depthSource.empty(); }
};

inline ErrCode
parseConfig(const std::string& content, const std::string& path, Config& config)
{
    DS3D_ERROR_RETURN(
        config::parseComponentConfig(content, path, config.compConfig),
        "parse depthsource component content failed");

    YAML::Node node = YAML::Load(config.compConfig.configBody);
    if (node["depth_source"]) {
        config.depthSource = node["depth_source"].as<std::string>();
        ;
    }
    if (node["color_source"]) {
        config.colorSource = node["color_source"].as<std::string>();
        ;
    }

    DS3D_FAILED_RETURN(
        !config.depthSource.empty() || !config.colorSource.empty(), ErrCode::kConfig,
        "depth_source and color_source, at least of them must be configured");

    if (node["depth_scale"]) {
        config.depthScale = node["depth_scale"].as<double>();
    }

    if (node["depth_datatype"]) {
        auto type = node["depth_datatype"].as<std::string>();
        DS3D_THROW_ERROR(
            type == "uint16" || type == "int16", ErrCode::kConfig, "depth_datatype must be uint16");
    }

    if (node["color"]) {
        auto color = node["color"].as<std::string>();
        DS3D_THROW_ERROR(color == "rgba", ErrCode::kConfig, "color must be rgba");
    }
    if (node["depth_size"]) {
        auto size = node["depth_size"].as<std::vector<uint32_t>>();
        DS3D_THROW_ERROR(size.size() == 2, ErrCode::kConfig, "depth_size must be [width, height]");
        config.depthPlane.width = size[0];
        config.depthPlane.height = size[1];
        config.depthPlane.bytesPerPixel = sizeof(uint16_t);
        config.depthPlane.pitchInBytes = config.depthPlane.width * config.depthPlane.bytesPerPixel;
    }

    if (node["color_size"]) {
        auto size = node["color_size"].as<std::vector<uint32_t>>();
        DS3D_THROW_ERROR(size.size() == 2, ErrCode::kConfig, "color_size must be [width, height]");
        config.colorPlane.width = size[0];
        config.colorPlane.height = size[1];
        config.colorPlane.bytesPerPixel = sizeof(vec4b);
        config.colorPlane.pitchInBytes = config.colorPlane.width * config.colorPlane.bytesPerPixel;
    }

    auto parseIntrinsic = [&](YAML::Node n, IntrinsicsParam& intrinsic) {
        DS_ASSERT(bool(n));
        intrinsic.width = n["width"].as<uint32_t>();
        intrinsic.height = n["height"].as<uint32_t>();
        intrinsic.centerX = n["centerX"].as<float>();
        intrinsic.centerY = n["centerY"].as<float>();
        intrinsic.fx = n["fx"].as<float>();
        intrinsic.fy = n["fy"].as<float>();
    };

    if (node["depth_intrinsic"]) {
        parseIntrinsic(node["depth_intrinsic"], config.depthIntrinsics);
        config.depthIntrinIsSet = true;
    }
    if (node["color_intrinsic"]) {
        parseIntrinsic(node["color_intrinsic"], config.colorIntrinsics);
        config.colorIntrinIsSet = true;
    }

    if (node["depth_to_color_extrinsic"]) {
        auto yExt = node["depth_to_color_extrinsic"];
        auto r = yExt["rotation"].as<std::vector<float>>();
        DS3D_THROW_ERROR(
            r.size() == 9, ErrCode::kConfig,
            "depth_to_color_extrinsic.rotation must be specified as 9 float values");
        array2Vec3<float>(&r[0], config.d2cExtrinsics.rotation[0]);
        array2Vec3<float>(&r[3], config.d2cExtrinsics.rotation[1]);
        array2Vec3<float>(&r[6], config.d2cExtrinsics.rotation[2]);
        auto tr = yExt["translation"].as<std::vector<float>>();
        DS3D_THROW_ERROR(
            tr.size() == 3, ErrCode::kConfig,
            "depth_to_color_extrinsic.translation must be specified as 3 float values");
        config.depth2ColorExSet = true;
    }
    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::depthsource

#endif  // _DS3D_DATALOADER_DEPTHSOURCE_CONFIG_H