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

#ifndef _DS3D_DATARENDER_GL3D_RENDER_CONFIG_H
#define _DS3D_DATARENDER_GL3D_RENDER_CONFIG_H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"

namespace ds3d { namespace impl { namespace render {

constexpr uint32_t kDefaultWidth = 1280;
constexpr uint32_t kDefaultHeight = 720;

enum class RenderType : int {
    kUnknown = 0,
    kTexture3D = 1,
    kLidar3D = 2,
    kPointCloudImage = 3,
    kDepthColor = 4,
};

using RenderItem = std::tuple<RenderType, std::string>;

/** @brief Ensemble render configuration.
 * Example:
 *   window_width: 1920
 *   window_height: 1080
 *   window_title: ds3d-gles-ensemble-render
 *   color_clear: true # true by default
 *   block: true # true by default
 *   render_graph: # multiple render context running in sequence order
 *     - texture3d_render:
 *        layout: [0, 0, 1920, 1080]
 *     - lidar3d_render:
 *        layout: [0, 0, 1920, 1080]
 *     - pointcloud_color_render:
 *        layout: [0, 0, 1920, 1080]
 */
struct EnsembleConfig {
    config::ComponentConfig compConfig;
    uint32_t winWidth = kDefaultWidth;
    uint32_t winHeight = kDefaultHeight;
    std::string winTitle = "ds3d-gles-ensemble-render";
    bool winClear = true;
    bool block = true;
    std::vector<RenderItem> ensembleContents;
};

inline RenderType
str2RenderType(const std::string& str)
{
    static const std::unordered_map<std::string, RenderType> kTypes{
        {"texture3d_render", RenderType::kTexture3D},
        {"lidar3d_render", RenderType::kLidar3D},
        {"pointcloud_color", RenderType::kPointCloudImage},
        {"depth_render", RenderType::kDepthColor},
    };
    auto iter = kTypes.find(str);
    if (iter == kTypes.cend()) {
        return RenderType::kUnknown;
    }
    return iter->second;
}

inline ErrCode
parseConfig(const std::string& content, const std::string& path, EnsembleConfig& config)
{
    DS3D_ERROR_RETURN(
        config::parseComponentConfig(content, path, config.compConfig),
        "parse dataloader component content failed");

    YAML::Node node = YAML::Load(config.compConfig.configBody);
    auto yWidth = node["window_width"];
    auto yHeight = node["window_height"];
    auto yTitle = node["window_title"];
    auto yBlock = node["block"];
    auto yColorClear = node["color_clear"];
    auto yRenders = node["render_graph"];

    if (yWidth) {
        config.winWidth = yWidth.as<uint32_t>();
    }
    if (yHeight) {
        config.winHeight = yHeight.as<uint32_t>();
    }

    if (yTitle) {
        config.winTitle = yTitle.as<std::string>();
    }
    if (yBlock) {
        config.block = yBlock.as<bool>();
    }
    if (yColorClear) {
        config.winClear = yColorClear.as<bool>();
    }
    DS3D_FAILED_RETURN(yRenders, ErrCode::kConfig, "No render_graph found.");

    for (auto render : yRenders) {
        for (const auto& kv : render) {
            std::string strType = kv.first.as<std::string>();
            YAML::Emitter renderContent;
            renderContent << kv.second;
            std::string strCont = cppString(renderContent.c_str(), renderContent.size());
            RenderType t = str2RenderType(strType);
            if (t == RenderType::kUnknown) {
                LOG_WARNING("unknown render type: %s, skipped", strType.c_str());
                continue;
            }
            config.ensembleContents.push_back(RenderItem{t, strCont});
        }
    }

    return ErrCode::kGood;
}


}}}  // namespace ds3d::impl::render

#endif  // _DS3D_DATARENDER_GL3D_RENDER_CONFIG_H