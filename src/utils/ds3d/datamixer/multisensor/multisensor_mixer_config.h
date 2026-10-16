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

#ifndef DS3D_DATAMIXER_MULTISENSOR_CONFIG_H
#define DS3D_DATAMIXER_MULTISENSOR_CONFIG_H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"

namespace ds3d { namespace impl { namespace mixer {

constexpr const float kMultiSensorMixerDefaultFrameRate = 30.0;
constexpr const double kSToNas = 1000000000.0;
constexpr const double kMsToNas = 1000000.0;
constexpr const double kUsToNas = 1000.0;
constexpr const double kNasToMs = (1.0 / 1000000.0);
constexpr const double kMultiSensorMixerDefaultFrameTimeout = kSToNas / kMultiSensorMixerDefaultFrameRate;
constexpr const bool kMultiSensorMixerDefaultForceSync = true;

struct Config {
    config::ComponentConfig compConfig;
    double frameTimeout = kMultiSensorMixerDefaultFrameTimeout;
    bool forceSync = kMultiSensorMixerDefaultForceSync;
};

inline ErrCode
parseConfig(const std::string& content, const std::string& path, Config& config)
{
    DS3D_ERROR_RETURN(
        config::parseComponentConfig(content, path, config.compConfig), "parse dataloader component content failed");
    YAML::Node node = YAML::Load(config.compConfig.configBody);
    auto yFrameTimeout = node["timeout"];
    auto yForceSync = node["force_sync"];

    if (yFrameTimeout) {
        config.frameTimeout = yFrameTimeout.as<double>() * kMsToNas;
    }
    if (yForceSync) {
        config.forceSync = yForceSync.as<bool>();
    }
    return ErrCode::kGood;
}


}}}  // namespace ds3d::impl::mixer

#endif  // DS3D_DATAMIXER_MULTISENSOR_CONFIG_H
