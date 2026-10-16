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

#ifndef DS3D_DATAFILTER_ALIGNMENT_CONFIG_H
#define DS3D_DATAFILTER_ALIGNMENT_CONFIG_H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"

namespace ds3d { namespace impl { namespace filter {

constexpr uint32_t kDefaultPointNum = 2048000;
constexpr uint32_t kDefaultPoolSize = 4;
constexpr bool kDefaultAlignToIntrinsic = true;
constexpr uint32_t kDefaultLidarElemSize = 4;
constexpr uint32_t kDefaultCamWidth = 0;
constexpr uint32_t kDefaultCamHeight = 0;

struct Config {
    config::ComponentConfig compConfig;
    std::vector<std::string> camInputKey;
    std::vector<std::string> lidarInputKey;
    std::vector<std::string> lidarAlignedKey;

    std::vector<std::string> outputCamIntrParmKeys{kCamIntrinsicParm};
    std::vector<std::string> outputCamIntrMatKeys{kCamIntrinsicMat};
    std::vector<std::string> outputLidarToCamExtrMatKeys{kLidarToCamExtrinsicMat};

    std::vector<IntrinsicsParam> camIntrinsicParm;
    std::vector<TransformMatrix> camIntrinsicMat;
    std::vector<TransformMatrix> lidarToCamExtrinsicMat;
    bool disableLidarDataAlign = false;  // If disabled, lidar data would bypass
    bool rowMajor =
        false;  // if matrix in config is row-major, need parse and convert into column-major
    uint32_t maxPoints = kDefaultPointNum;
    uint32_t memPoolSize = kDefaultPoolSize;
    bool alignToIntrinsic = kDefaultAlignToIntrinsic;
    uint32_t lidarElemSize = kDefaultLidarElemSize;
    uint32_t camWidth = kDefaultCamWidth;
    uint32_t camHeight = kDefaultCamHeight;
};

inline ErrCode
readTransformMatrix(const std::vector<float>& vec12, bool rowMajor, TransformMatrix& transform)
{
    DS3D_FAILED_RETURN(
        vec12.size() == 12, ErrCode::kConfig,
        "config matrix is a vector with comma separated floats representing 3x4 or 4x3 matrix in "
        "%s-major",
        (rowMajor ? "row" : "column"));

    if (rowMajor) {
        for (int i = 0; i < 3; i++) {
            transform.matrix[i] = {{vec12[i], vec12[i + 4], vec12[i + 8], 0}};
        }
        transform.matrix[3] = {{vec12[3], vec12[3 + 4], vec12[3 + 8], 1}};
    } else {  // column major order
        for (int i = 0; i < 3; i++) {
            transform.matrix[i] = {{vec12[i * 3], vec12[i * 3 + 1], vec12[i * 3 + 2], 0}};
        }
        transform.matrix[3] = {{vec12[9], vec12[10], vec12[11], 1}};
    }
    return ErrCode::kGood;
}

/**
 * The calib file parsed here have data in row major order
 * We read it into column major order
 */
inline ErrCode
parseConfig(const std::string& content, const std::string& path, Config& config)
{
    DS3D_ERROR_RETURN(
        config::parseComponentConfig(content, path, config.compConfig), "parse dataloader component content failed");

    YAML::Node node = YAML::Load(config.compConfig.configBody);
    auto yCamInputKey = node["cam_input_keys"];
    auto yLidarInputKey = node["lidar_input_keys"];
    auto yLidarAlignedKey = node["lidar_aligned_keys"];
    auto yCamIntrinsic = node["cam_intrinsic"];
    auto yLidarToCamExtrinsic = node["lidar_to_cam_extrisic"];
    auto yMaxPoints = node["max_points"];
    auto yMemPoolSize = node["mem_pool_size"];
    auto yAlignToIntrinsic = node["align_to_intrinsic"];
    auto yLidarElemSize = node["lidar_element_size"];
    auto yCamWidth = node["cam_width"];
    auto yCamHeight = node["cam_height"];
    auto yDisableLidarDataAlign = node["disable_lidar_data_align"];
    auto yRowMajor = node["row_major"];

    auto yOutputCamIntrParmKeys = node["output_cam_intrinsic_param_keys"];
    auto yOutputCamIntrMatKeys = node["output_cam_intrinsic_mat_keys"];
    auto yOutputLidarToCamExtrMatKeys = node["output_lidar_to_cam_extrinsic_mat_keys"];


    if (yOutputCamIntrParmKeys) {
        config.outputCamIntrParmKeys = yOutputCamIntrParmKeys.as<std::vector<std::string>>();
    }
    if (yOutputCamIntrMatKeys) {
        config.outputCamIntrMatKeys = yOutputCamIntrMatKeys.as<std::vector<std::string>>();
    }
    if (yOutputLidarToCamExtrMatKeys) {
        config.outputLidarToCamExtrMatKeys =
            yOutputLidarToCamExtrMatKeys.as<std::vector<std::string>>();
    }

    if (yRowMajor) {
        config.rowMajor = yRowMajor.as<bool>();
    }
    if (yDisableLidarDataAlign) {
        config.disableLidarDataAlign = yDisableLidarDataAlign.as<bool>();
    }
    if (yCamInputKey) {
        config.camInputKey = yCamInputKey.as<std::vector<std::string>>();
    }
    if (yLidarInputKey) {
        config.lidarInputKey = yLidarInputKey.as<std::vector<std::string>>();
    }
    if (yLidarAlignedKey) {
        config.lidarAlignedKey = yLidarAlignedKey.as<std::vector<std::string>>();
    }
    if (yCamIntrinsic) {
        std::vector<std::vector<float>> vCamIntrinsic;
        if (yCamIntrinsic.begin()->Type() == YAML::NodeType::Sequence) {  // list of list
            for (auto yEachMatrix : yCamIntrinsic) {
                auto vec12 = yEachMatrix.as<std::vector<float>>();
                vCamIntrinsic.emplace_back(vec12);
            }
        } else {  // list of scalar
            vCamIntrinsic.resize(1);
            vCamIntrinsic[0] = yCamIntrinsic.as<std::vector<float>>();
        }

        uint32_t numCam = vCamIntrinsic.size();
        config.camIntrinsicMat.resize(numCam);
        config.camIntrinsicParm.resize(numCam);
        for (uint32_t i = 0; i < numCam; ++i) {
            DS3D_ERROR_RETURN(
                readTransformMatrix(vCamIntrinsic[i], config.rowMajor, config.camIntrinsicMat[i]),
                "cam_intrinsic[%d] parse failed.", i);

            config.camIntrinsicParm[i].centerX = config.camIntrinsicMat[i].matrix[2].data[0];
            config.camIntrinsicParm[i].centerY = config.camIntrinsicMat[i].matrix[2].data[1];
            config.camIntrinsicParm[i].fx = config.camIntrinsicMat[i].matrix[0].data[0];
            config.camIntrinsicParm[i].fy = config.camIntrinsicMat[i].matrix[1].data[1];

            LOG_DEBUG(
                "config.camIntrinsicParm[%d]; [centerX,Y=(%f,%f); [fx,fy=(%f,%f)]; "
                "[w,h=(%d,%d)]\n",
                i, config.camIntrinsicParm[i].centerX, config.camIntrinsicParm[i].centerY,
                config.camIntrinsicParm[i].fx, config.camIntrinsicParm[i].fy,
                config.camIntrinsicParm[i].width, config.camIntrinsicParm[i].height);
        }
    }
    if (yLidarToCamExtrinsic) {
        std::vector<std::vector<float>> vLidarToCamExtrinsic;
        if (yLidarToCamExtrinsic.begin()->Type() == YAML::NodeType::Sequence) {  // list of list
            for (auto yEachMatrix : yLidarToCamExtrinsic) {
                auto vec12 = yEachMatrix.as<std::vector<float>>();
                vLidarToCamExtrinsic.emplace_back(vec12);
            }
        } else {  // list of scalar
            vLidarToCamExtrinsic.resize(1);
            vLidarToCamExtrinsic[0] = yLidarToCamExtrinsic.as<std::vector<float>>();
        }

        uint32_t numLidar = vLidarToCamExtrinsic.size();
        config.lidarToCamExtrinsicMat.resize(numLidar);
        for (uint32_t i = 0; i < numLidar; ++i) {
            DS3D_ERROR_RETURN(
                readTransformMatrix(
                    vLidarToCamExtrinsic[i], config.rowMajor, config.lidarToCamExtrinsicMat[i]),
                "lidar_to_cam_extrisic[%d] parse failed.", i);
        }
    }
    if (yMaxPoints) {
        config.maxPoints = yMaxPoints.as<uint32_t>();
    }
    if (yMemPoolSize) {
        config.memPoolSize = yMemPoolSize.as<uint32_t>();
    }
    if (yAlignToIntrinsic) {
        config.alignToIntrinsic = yAlignToIntrinsic.as<bool>();
    }
    if (yLidarElemSize) {
        config.lidarElemSize = yLidarElemSize.as<uint32_t>();
        DS3D_FAILED_RETURN(
            ((config.lidarElemSize == 4) || (config.lidarElemSize == 3)), ErrCode::kConfig,
            "lidar_element_size invalid value; only 3 or 4 supported");
    }
    if (yCamWidth) {
        config.camWidth = yCamWidth.as<uint32_t>();
        for (auto& camParam : config.camIntrinsicParm) {
            camParam.width = config.camWidth;
        }
    }
    if (yCamHeight) {
        config.camHeight = yCamHeight.as<uint32_t>();
        for (auto& camParam : config.camIntrinsicParm) {
            camParam.height = config.camHeight;
        }
    }
    return ErrCode::kGood;
}


}}}  // namespace ds3d::impl::filter

#endif  // DS3D_DATAFILTER_ALIGNMENT_CONFIG_H
