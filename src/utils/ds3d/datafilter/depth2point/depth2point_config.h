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

#ifndef DS3D_DATAFILTER_DEPTH_2_POINT_CONFIG_H
#define DS3D_DATAFILTER_DEPTH_2_POINT_CONFIG_H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"

namespace ds3d { namespace impl { namespace filter {

constexpr uint32_t kDefaultWidth = 1280;
constexpr uint32_t kDefaultHeight = 720;
constexpr float kMinDepth = 0.3f;
constexpr float kMaxDepth = 5.0f;

constexpr const char* kDepthStream = "depth";
constexpr const char* kColorStream = "color";
constexpr const char* kPointsStream = "points";

struct Config {
    config::ComponentConfig compConfig;
    std::set<std::string> inStreams;
    uint32_t maxPoints = 1920 * 1080;
    uint32_t memPoolSize = 8;

    bool hasDepthStream() const { return inStreams.count(kDepthStream); }
    bool hasColorStream() const { return inStreams.count(kColorStream); }

    bool check()
    {
        if (!maxPoints || !memPoolSize) {
            LOG_ERROR("depth2points configed wrong max_points or mem_pool_size");
            return false;
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
    auto streams = node["in_streams"];
    auto yMaxPoints = node["max_points"];
    auto yMemPoolSize = node["mem_pool_size"];

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
            config.inStreams.emplace(each);
        }
    }
    if (yMaxPoints) {
        config.maxPoints = yMaxPoints.as<uint32_t>();
    }
    if (yMemPoolSize) {
        config.memPoolSize = yMemPoolSize.as<uint32_t>();
    }
    return ErrCode::kGood;
}


}}}  // namespace ds3d::impl::filter

#endif  // DS3D_DATAFILTER_DEPTH_2_POINT_CONFIG_H