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

#ifndef _DS3D_DATARENDER_GL3D_RENDER_CONFIG__H
#define _DS3D_DATARENDER_GL3D_RENDER_CONFIG__H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"

namespace ds3d { namespace impl { namespace render {

constexpr uint32_t kDefaultWidth = 1280;
constexpr uint32_t kDefaultHeight = 720;
constexpr float kMinDepth = 0.3f;
constexpr float kMaxDepth = 5.0f;

constexpr const char* kDepthStream = "depth";
constexpr const char* kColorStream = "color";
constexpr const char* kPointStream = "points";
constexpr const char* kLidarStream = "lidardata";

static vec3b kDefaultMinDepthColor{{255, 255, 0}};
static vec3b kDefaultMaxDepthColor{{0, 0, 0}};

struct Config {
    config::ComponentConfig compConfig;
    std::set<std::string> streams;
    uint32_t width = kDefaultWidth;
    uint32_t height = kDefaultHeight;
    std::string title = "data-render";
    bool block = false;

    bool hasDepthStream() const { return streams.count(kDepthStream); }
    bool hasColorStream() const { return streams.count(kColorStream); }
    bool hasPointsStream() const { return streams.count(kPointStream); }

    bool check()
    {
        if (streams.empty()) {
            streams.emplace(kDepthStream);
            streams.emplace(kColorStream);
        }
        if (!width) {
            width = kDefaultWidth;
        }
        if (!height) {
            width = kDefaultHeight;
        }

        return true;
    }
};

struct DepthConfig : public Config {
    float minDepth = kMinDepth;
    float maxDepth = kMaxDepth;
    vec3b minDepthColor = kDefaultMinDepthColor;
    vec3b maxDepthColor = kDefaultMaxDepthColor;
};

struct PointsConfig : public Config {
    vec3f viewPosition{{0.0f, 0.0f, 0.0f}};
    vec3f viewTarget{{0.0f, 0.0f, -1.0f}};
    vec3f viewUp{{0.0f, 1.0f, 0.0f}};
    float near = 0.3f;
    float far = 10.0f;
    float fov = 60.0f;  // angle degree
    bool coordYOpposite = true;
    bool positiveZOnly = false;
};

struct LidarConfig : public PointsConfig {
    uint32_t elementSize = 4;
    std::string lidarDataKey = kLidarXYZI;
    std::string lidarBboxKey = kLidar3DBboxRawData;
    vec3b lidarColor = {{0, 255, 0}};  // RGB
    bool enableLabel = false;
};

inline ErrCode
parseBaseConfig(const std::string& content, const std::string& path, Config& config)
{
    DS3D_ERROR_RETURN(
        config::parseComponentConfig(content, path, config.compConfig),
        "parse dataloader component content failed");

    YAML::Node node = YAML::Load(config.compConfig.configBody);
    auto streams = node["streams"];
    auto yWidth = node["width"];
    auto yHeight = node["height"];
    auto yTitle = node["title"];
    auto yBlock = node["block"];

    if (streams) {
        const static std::set<std::string> kValidStreams{
            kDepthStream, kColorStream, kPointStream, kLidarStream};
        auto toString = [&]() {
            std::string r("[");
            for (auto& i : kValidStreams) {
                r += i + ",";
            }
            return r + "]";
        };
        std::vector<std::string> strStreams = streams.as<std::vector<std::string>>();
        for (auto& each : strStreams) {
            DS3D_THROW_ERROR(
                kValidStreams.count(each), ErrCode::kConfig,
                "stream must choose from " + toString());
            config.streams.emplace(each);
        }
    }
    if (yWidth) {
        config.width = yWidth.as<uint32_t>();
    }
    if (yHeight) {
        config.height = yHeight.as<uint32_t>();
    }

    if (yTitle) {
        config.title = yTitle.as<std::string>();
    }
    if (yBlock) {
        config.block = yBlock.as<bool>();
    }
    return ErrCode::kGood;
}

inline ErrCode
parseDepthColorConfig(const std::string& content, const std::string& path, DepthConfig& config)
{
    DS3D_ERROR_RETURN(
        parseBaseConfig(content, path, config), "paser datarender config base failed.");
    YAML::Node node = YAML::Load(config.compConfig.configBody);
    auto yminDepth = node["min_depth"];
    auto ymaxDepth = node["max_depth"];
    auto yMinDepthColor = node["min_depth_color"];
    auto yMaxDepthColor = node["max_depth_color"];

    if (yminDepth) {
        config.minDepth = yminDepth.as<float>();
    }
    if (ymaxDepth) {
        config.maxDepth = ymaxDepth.as<float>();
    }
    if (yMinDepthColor) {
        auto color = yMinDepthColor.as<std::vector<uint32_t>>();
        DS3D_FAILED_RETURN(
            color.size() >= 3, ErrCode::kConfig, "min_depth_color need RGB 3 values");
        config.minDepthColor.x = color[0];
        config.minDepthColor.y = color[1];
        config.minDepthColor.z = color[2];
    }
    if (yMaxDepthColor) {
        auto color = yMaxDepthColor.as<std::vector<uint32_t>>();
        DS3D_FAILED_RETURN(
            color.size() >= 3, ErrCode::kConfig, "max_depth_color need RGB 3 values");
        config.maxDepthColor.x = color[0];
        config.maxDepthColor.y = color[1];
        config.maxDepthColor.z = color[2];
    }
    return ErrCode::kGood;
}

inline ErrCode
parsePointsConfig(const std::string& content, const std::string& path, PointsConfig& config)
{
    DS3D_ERROR_RETURN(
        parseBaseConfig(content, path, config), "paser datarender config base failed.");
    YAML::Node node = YAML::Load(config.compConfig.configBody);
    auto yViewPos = node["view_position"];
    auto yViewTarget = node["view_target"];
    auto yViewUp = node["view_up"];
    auto yNear = node["near"];
    auto yFar = node["far"];
    auto yFov = node["fov"];
    auto yCoordYOpp = node["coord_y_opposite"];
    auto yPositiveZonly = node["positive_z_only"];

    if (yViewPos) {
        auto pos = yViewPos.as<std::vector<float>>();
        DS3D_FAILED_RETURN(pos.size() >= 3, ErrCode::kConfig, "view_position need x,y,z 3 values");
        config.viewPosition.x = pos[0];
        config.viewPosition.y = pos[1];
        config.viewPosition.z = pos[2];
    }
    if (yViewTarget) {
        auto target = yViewTarget.as<std::vector<float>>();
        DS3D_FAILED_RETURN(target.size() >= 3, ErrCode::kConfig, "view_target need x,y,z 3 values");
        config.viewTarget.x = target[0];
        config.viewTarget.y = target[1];
        config.viewTarget.z = target[2];
    }
    if (yViewUp) {
        auto up = yViewUp.as<std::vector<float>>();
        DS3D_FAILED_RETURN(up.size() >= 3, ErrCode::kConfig, "view_up need x,y,z 3 values");
        config.viewUp.x = up[0];
        config.viewUp.y = up[1];
        config.viewUp.z = up[2];
    }
    if (yNear) {
        config.near = yNear.as<float>();
    }
    if (yFar) {
        config.far = yFar.as<float>();
    }
    if (yFov) {
        config.fov = yFov.as<float>();
    }
    if (yCoordYOpp) {
        config.coordYOpposite = yCoordYOpp.as<bool>();
    }
    if (yPositiveZonly) {
        config.positiveZOnly = yPositiveZonly.as<bool>();
    }
    return ErrCode::kGood;
}

inline ErrCode
parseLidarConfig(const std::string& content, const std::string& path, LidarConfig& config)
{
    DS3D_ERROR_RETURN(
        parsePointsConfig(content, path, config), "paser datarender config base failed.");
    YAML::Node node = YAML::Load(config.compConfig.configBody);
    auto yEleSize = node["element_size"];
    auto yLidarDataKey = node["lidar_data_key"];
    auto yLidarBboxKey = node["lidar_bbox_key"];
    auto yColor = node["lidar_color"];
    auto ylabel = node["enable_label"];

    if (yEleSize) {
        config.elementSize = yEleSize.as<uint32_t>();
    }
    if (yLidarDataKey) {
        config.lidarDataKey = yLidarDataKey.as<std::string>();
    }
    if (yLidarBboxKey) {
        config.lidarBboxKey = yLidarBboxKey.as<std::string>();
    }
    if (yColor) {
        auto color = yColor.as<std::vector<uint32_t>>();
        size_t size = (color.size() < 3 ? color.size() : 3);
        for (size_t i = 0; i < size; ++i) {
            config.lidarColor.data[i] = (uint8_t)color[i];
        }
    }
    if (ylabel) {
        config.enableLabel = ylabel.as<bool>();
    }

    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::render

#endif  // _DS3D_DATARENDER_GL3D_RENDER_CONFIG__H