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

#ifndef DS3D_DATABRIDGE_VIDEO_CONFIG_H
#define DS3D_DATABRIDGE_VIDEO_CONFIG_H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"

namespace ds3d { namespace impl { namespace bridge {

struct Config {
    config::ComponentConfig compConfig;
    bool surfaceToImage = false;
    std::string outputImageKey = kColorFrame;
    std::string outputObject2DKey = kObject2DBboxKey;
    std::string outputVideoPreProcTensorKey = kVideoPreProcTensor;
};

inline ErrCode
parseConfig(const std::string& content, const std::string& path, Config& config)
{
    DS3D_ERROR_RETURN(
        config::parseComponentConfig(content, path, config.compConfig),
        "parse dataloader component content failed");

    YAML::Node node = YAML::Load(config.compConfig.configBody);
    auto ySurf2Img = node["surface_to_image"];
    auto yImageKey = node["output_image_key"];
    auto yObj2DKey = node["output_object2d_key"];
    auto yVideoPreProcKey = node["output_video_preprocess_key"];
    if (ySurf2Img) {
        config.surfaceToImage = ySurf2Img.as<bool>();
    }
    if (yImageKey) {
        config.outputImageKey = yImageKey.as<std::string>();
    }
    if (yObj2DKey) {
        config.outputObject2DKey = yObj2DKey.as<std::string>();
    }
    if (yVideoPreProcKey) {
        config.outputVideoPreProcTensorKey = yVideoPreProcKey.as<std::string>();
    }

    return ErrCode::kGood;
}


}}}  // namespace ds3d::impl::bridge

#endif  // DS3D_DATABRIDGE_VIDEO_CONFIG_H
