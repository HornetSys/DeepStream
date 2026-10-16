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

#ifndef _DS3D_DATARENDER_LIDAR_FILE_RENDER_CONFIG__H
#define _DS3D_DATARENDER_LIDAR_FILE_RENDER_CONFIG__H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"

#define _PATH_MAX 4096

namespace ds3d { namespace impl { namespace lidarrender {

struct Config {
    config::ComponentConfig compConfig;
    std::string lidarFrameSavePath = "";
    std::string inputDatamapKey = "";
};


inline ErrCode
parseBaseConfig(const std::string& content, const std::string& path, Config& config)
{
    DS3D_ERROR_RETURN(
        config::parseComponentConfig(content, path, config.compConfig),
        "parse dataloader component content failed");
    YAML::Node node = YAML::Load(config.compConfig.configBody);
    char absRealFilePath[_PATH_MAX + 1];
    if (node["frames_save_path"]) {
        config.lidarFrameSavePath = node["frames_save_path"].as<std::string>();
        if(config.lidarFrameSavePath[0] != '/') {
            if (!realpath(path.c_str(), absRealFilePath)) {
                if (errno != ENOENT) {
                    LOG_WARNING("Your config file path is not right!");
                    return ErrCode::kNotFound;
                }
            }
            std::string tmpPath = absRealFilePath;
            int pos = tmpPath.find_last_of("/");
            config.lidarFrameSavePath = tmpPath.substr(0, pos+1) + config.lidarFrameSavePath;
            if(config.lidarFrameSavePath.back() != '/')
                config.lidarFrameSavePath += '/';
        }
        LOG_DEBUG("config path %s",config.lidarFrameSavePath.c_str());
    }

    if(node["input_datamap_key"]) {
        config.inputDatamapKey = node["input_datamap_key"].as<std::string>();
    } else {
        config.inputDatamapKey = kLidar3DBboxRawData;
    }

    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::lidarrender

#endif  // _DS3D_DATARENDER_LIDAR_FILE_RENDER_CONFIG__H