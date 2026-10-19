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

#ifndef _DS3D_DATALOADER_REALSENSE_CONFIG__H
#define _DS3D_DATALOADER_REALSENSE_CONFIG__H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"

namespace ds3d { namespace impl { namespace realsense {

constexpr const char* kDepthStream = "depth";
constexpr const char* kColorStream = "color";
constexpr const char* kPointsStream = "points";

struct Config {
    config::ComponentConfig compConfig;
    std::set<std::string> streams;
    uint32_t framerate = 0;
    bool alignImage2Depth = false;

    bool hasDepthStream() const { return streams.count(kDepthStream); }
    bool hasColorStream() const { return streams.count(kColorStream); }
    bool hasPointsStream() const { return streams.count(kPointsStream); }
    bool check()
    {
        if (streams.empty()) {
            streams.emplace(kDepthStream);
            streams.emplace(kColorStream);
        }
        return true;
    }
};

inline ErrCode
parseConfig(const std::string& content, const std::string& path, Config& config)
{
    DS3D_ERROR_RETURN(
        config::parseComponentConfig(content, path, config.compConfig),
        "parse dataloader component content failed");

    YAML::Node node = YAML::Load(config.compConfig.configBody);
    auto streams = node["streams"];
    auto framerate = node["framerate"];
    auto alignedImage2Depth = node["aligned_image_to_depth"];

    if (streams) {
        const static std::set<std::string> kValidStreams{kDepthStream, kColorStream, kPointsStream};
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
    if (framerate) {
        uint32_t fr = framerate.as<uint32_t>();
        config.framerate = fr;
    }
    if (alignedImage2Depth) {
        config.alignImage2Depth = alignedImage2Depth.as<bool>();
    }
    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::realsense

#endif  // _DS3D_DATALOADER_REALSENSE_CONFIG__H