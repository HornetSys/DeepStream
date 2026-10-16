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

#ifndef DS3D_DATAFILTER_LIDAR_INFERENCE_CONFIG_H
#define DS3D_DATAFILTER_LIDAR_INFERENCE_CONFIG_H

#include <ds3d/common/common.h>
#include <ds3d/common/func_utils.h>

#include "ds3d/common/hpp/yaml_config.hpp"

#include "infer_utils.h"

using namespace nvdsinferserver;


namespace ds3d { namespace impl { namespace filter {


constexpr const char* kLidarStream = "lidar";
constexpr const char* kVideoStream = "video";
constexpr const char* kDS3DStream = "ds3d";

struct ModelInputDesc : public InferBufferDescription {
    std::string fromKey;
    bool is2DFrame = false;
};

struct Config {
    config::ComponentConfig compConfig;
    std::set<std::string> inStreams;
    std::string filterInputDatamapKey;
    uint32_t memPoolSize = 8;
    std::map<size_t, std::string> modelInputs;
    //nvinferserver config file.
    std::string nvinferserverCfg;
    uint gpuid;
    InferMemType inputTensorMemType = InferMemType::kGpuCuda;
    std::map<std::string, ModelInputDesc> inputLayersDes;
    /** custom preprocess lib handle */
    void* customPreprocessLibHandle;
    std::string customPreprocessFuncName;
    //app's cfg path
    std::string configPath;

    bool hasLidarStream() const { return inStreams.count(kLidarStream); }

    bool check()
    {
        if (!memPoolSize) {
            LOG_ERROR("lidarinference configed wrong  mem_pool_size");
            return false;
        }
        return true;
    }
};

static bool isAbsolutePath(const std::string& path);

bool
isAbsolutePath(const std::string& path)
{
    return !path.empty() && (path[0] == '/');
}

inline ErrCode
parseConfig(const std::string& content, const std::string& path, Config& config)
{
    DS3D_ERROR_RETURN(
        config::parseComponentConfig(content, path, config.compConfig),
        "parse dataloader component content failed");
    YAML::Node node = YAML::Load(config.compConfig.configBody);
    auto streams = node["in_streams"];
    auto filterInputDatamapKey = node["filter_input_datamap_key"];
    auto yMemPoolSize = node["mem_pool_size"];
    auto modelInputs = node["model_inputs"];
    auto gpuid = node["gpu_id"];
    auto inputTensorMemType = node["input_tensor_mem_type"];
    auto customPreprocessLibPath = node["custom_preprocess_lib_path"];
    auto customPreprocessFuncName = node["custom_preprocess_func_name"];
    auto nvinferserverCfg = node["config_file"];
    auto configPath = node["config_path"];

    if (streams) {
        const static std::set<std::string> kValidStreams{kLidarStream, kVideoStream, kDS3DStream};
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
    if (filterInputDatamapKey) {
        config.filterInputDatamapKey =  filterInputDatamapKey.as<std::string>();
    }
    if (yMemPoolSize) {
        config.memPoolSize = yMemPoolSize.as<uint32_t>();
    }
    if (modelInputs) {
        for (auto item :modelInputs)
        {
            ModelInputDesc des;
            if(item["name"])
                des.name = item["name"].as<std::string>();  // name
            std::string dataType;
            if(item["datatype"]){
                dataType = item["datatype"].as<std::string>();
                des.dataType = grpcStr2DataType(dataType);  //datatype
            }
            std::vector<std::string> strStreams = item["shape"].as<std::vector<std::string>>();
            des.dims.numDims = strStreams.size();
            for(size_t i = 0; i < des.dims.numDims; i++){
                des.dims.d[i] = std::stoi(strStreams[i]);
            }
            des.dims.numElements = std::accumulate(des.dims.d, des.dims.d + des.dims.numDims, 1,
                [](int s, int i) { return s * i; });
            if (item["from"]) {
                des.fromKey = item["from"].as<std::string>();
            }
            if (item["is_2d_frame"]) {
                des.is2DFrame = item["is_2d_frame"].as<bool>();
            }
            LOG_INFO(
                "modelInputs name:%s, dataType:%s, ndataType:%d, numDims:%d, numElements:%d, from: "
                "%s",
                des.name.c_str(), dataType.c_str(), (int)des.dataType, des.dims.numDims,
                des.dims.numElements, des.fromKey.c_str());
            config.inputLayersDes.emplace(des.name, des);
        }
    }
    if (gpuid) {
        config.gpuid = gpuid.as<uint>();
    }
    if (inputTensorMemType) {
        std::string value = inputTensorMemType.as<std::string>();
        if(!value.compare("GpuCuda")){
            config.inputTensorMemType = InferMemType::kGpuCuda;
        } else if (!value.compare("CpuCuda") || !value.compare("Cpu")) {
            config.inputTensorMemType = InferMemType::kCpuCuda;
        } else {
            LOG_WARNING("%s not supported, use defualt value GpuCuda", value.c_str());
        }
    }
    if (nvinferserverCfg) {
        config.nvinferserverCfg = nvinferserverCfg.as<std::string>();
    }
    if (configPath) {
        config.configPath = configPath.as<std::string>();
    }
    if (customPreprocessLibPath) {
        std::string liPath = customPreprocessLibPath.as<std::string>();
        LOG_INFO("customPreprocessLibPath:%s", liPath.c_str());

        /* Initialize custom library */
        if (!liPath.empty()) {
            config.customPreprocessLibHandle = dlopen(liPath.c_str(), RTLD_NOW);
            if (!config.customPreprocessLibHandle) {
                LOG_INFO("dlopen failed, liPath:%s", liPath.c_str());
            } else {
                if (customPreprocessFuncName) {
                    config.customPreprocessFuncName = customPreprocessFuncName.as<std::string>();
                    if (config.customPreprocessFuncName.empty()) {
                        LOG_INFO("cant get customPreprocessFuncName");
                    }
                }
            }
        }
    }

    LOG_INFO("memPoolSize: %d", config.memPoolSize);
    LOG_INFO("gpuid: %d", config.gpuid);
    LOG_INFO("filterInputDatamapKey: %s", config.filterInputDatamapKey.c_str());
    LOG_INFO("inputTensorMemType: %d", (int)config.inputTensorMemType);
    LOG_INFO("customPreprocessLibHandle: %p", config.customPreprocessLibHandle);
    LOG_INFO("customPreprocessFuncName: %s", config.customPreprocessFuncName.c_str());
    LOG_INFO("nvinferserverCfg: %s", config.nvinferserverCfg.c_str());
    LOG_INFO("configPath: %s", config.configPath.c_str());

    //config data checking
    if(config.inputLayersDes.size() == 0){
       LOG_ERROR("model_inputs parse failed, abort");
       abort();
    }

    if(!isAbsolutePath(config.nvinferserverCfg)){
        size_t pos = config.configPath.find_last_of("/");
        if(pos != std::string::npos){
            std::string tmpPath = config.configPath.substr(0, pos+1);
            config.nvinferserverCfg = tmpPath + config.nvinferserverCfg;
            LOG_INFO("new nvinferserverCfg: %s", config.nvinferserverCfg.c_str());
        }
    }

    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::filter

#endif  // DS3D_DATAFILTER_LIDAR_INFERENCE_CONFIG_H
