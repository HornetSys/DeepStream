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

#ifndef DS3D_DATAFILTER_SENSOR_FUSIOIN_CONFIG_H
#define DS3D_DATAFILTER_SENSOR_FUSIOIN_CONFIG_H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"

#include <unordered_set>

namespace ds3d { namespace impl { namespace filter {

constexpr const float kDefaultIouThreshold = 0.1f;

struct MatchLabel {
    std::string newLabel;
    vec4f color = {{1.0f, 1.0f, 0, 1.0f}};  // RGBA;
    std::string label2D;
    std::string label3D;
};

using FusionLabels = std::vector<MatchLabel>;

struct Config {
    config::ComponentConfig compConfig;
    std::string input2DBboxKey = kObject2DBboxKey;
    std::string input3DbboxKey = kLidar3DBboxRawData;
    std::string input2DImageKey;  // empty means no input color image
    std::string camIntrinsicsKey = kCamIntrinsicMat;
    std::string camIntrinsicsParamKey = kCamIntrinsicParm;
    std::string lidar2CamExtrinsicsKey = kLidarToCamExtrinsicMat;
    std::string outputFusedKey = kFusedDetectionKey;
    std::string outputFusedBboxRenderKey;  // disable if no value set from config file, this is for
                                           // render purpose
    float iouThreshold = kDefaultIouThreshold;
    FusionLabels fusionLabels;
};

inline ErrCode
parseConfig(const std::string& content, const std::string& path, Config& config)
{
    DS3D_ERROR_RETURN(
        config::parseComponentConfig(content, path, config.compConfig),
        "parse dataloader component content failed");
    YAML::Node node = YAML::Load(config.compConfig.configBody);
    auto yInput2DBbox = node["2d_objects_input_key"];
    auto yInput3Dbbox = node["3d_objects_input_key"];
    auto y2dInputColor = node["2d_color_image_input_key"];
    auto yMatchClasses = node["match_classes"];
    auto yCamInt = node["cam_intrinsics_mat_key"];
    auto yCamIntParam = node["cam_intrinsics_param_key"];
    auto yLidar2CamExt = node["lidar_to_cam_extrinsics_mat_key"];
    auto yIouThreshold = node["iou_threshold"];
    auto yOutputFusedKey = node["output_fused_detection_key"];
    auto yOutputFusedRenderKey = node["output_fused_3d_bbox_key"];

    DS3D_FAILED_RETURN(
        yMatchClasses, ErrCode::kConfig, "<match_classes> list is required in sensorfusion config");

    std::unordered_set<std::string> labelsCheck;
    for (std::size_t i = 0; i < yMatchClasses.size(); i++) {
        const YAML::Node& kvs = yMatchClasses[i];
        for (const auto& kv : kvs) {
            auto yMatch = kv.second;
            auto y2D = yMatch["label_2d"];
            auto y3D = yMatch["label_3d"];
            auto yColor = yMatch["color"];

            MatchLabel match;
            match.newLabel = kv.first.as<std::string>();
            match.label2D = y2D.as<std::string>();
            match.label3D = y3D.as<std::string>();
            if (yColor) {
                auto color = yColor.as<std::vector<float>>();
                size_t size = (color.size() < 3 ? color.size() : 3);
                for (size_t i = 0; i < size; ++i) {
                    match.color.data[i] = color[i] / 255.0f;
                }
            }
            if (labelsCheck.count(match.newLabel)) {
                LOG_WARNING(
                    "new label: %s exist multiple times in sensor fusion config, it might match "
                    "multiple labels",
                    match.newLabel.c_str());
            } else {
                labelsCheck.insert(match.newLabel);
            }

            config.fusionLabels.emplace_back(match);
        }
    }

    if (yInput2DBbox) {
        config.input2DBboxKey = yInput2DBbox.as<std::string>();
    }
    if (yInput3Dbbox) {
        config.input3DbboxKey = yInput3Dbbox.as<std::string>();
    }
    if (y2dInputColor) {
        config.input2DImageKey = y2dInputColor.as<std::string>();
    }
    if (yCamInt) {
        config.camIntrinsicsKey = yCamInt.as<std::string>();
    }
    if (yCamIntParam) {
        config.camIntrinsicsParamKey = yCamIntParam.as<std::string>();
    }
    if (yLidar2CamExt) {
        config.lidar2CamExtrinsicsKey = yLidar2CamExt.as<std::string>();
    }
    if (yIouThreshold) {
        config.iouThreshold = yIouThreshold.as<float>();
    }
    if (yOutputFusedKey) {
        config.outputFusedKey = yOutputFusedKey.as<std::string>();
    }
    if (yOutputFusedRenderKey) {
        config.outputFusedBboxRenderKey = yOutputFusedRenderKey.as<std::string>();
    }

    LOG_DEBUG("fusion config, 2d_objects_input_key: %s", config.input2DBboxKey.c_str());
    LOG_DEBUG("fusion config, 3d_objects_input_key: %s", config.input3DbboxKey.c_str());
    std::stringstream ssKeys;
    for (const auto& label : config.fusionLabels) {
        ssKeys << " " << label.newLabel << ",";
    }
    LOG_DEBUG("fusion config, 2d_input_key:%s", ssKeys.str().c_str());

    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::filter

#endif  // DS3D_DATAFILTER_SENSOR_FUSIOIN_CONFIG_H
