/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "engine_builder.h"
#include <memory>
#include <vector>
#include <sys/stat.h>
#include <string>
#include <iostream>
#include <fstream>

// Function to read the entire file into a vector - marked static to limit scope to this file
static std::vector<char> readFile(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return {};
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Error reading file: " << filePath << std::endl;
        return {};
    }

    return buffer;
}

bool buildEngineFromOnnx(
    const std::string& onnxFilePath,
    const std::string& engineFilePath,
    int batchSize,
    int numSensors,
    bool fp16Mode,
    PrettyLogger* logger
) {
    // Create TensorRT logger with ERROR severity to hide verbose reshape messages
    static Logger trtLogger(nvinfer1::ILogger::Severity::kERROR);

    // Create builder
    TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(trtLogger));
    if (!builder) {
        logger->error("Failed to create TensorRT builder");
        return false;
    }

    // Create network definition - in TensorRT 10.3, we don't use kEXPLICIT_BATCH flag
    // as it's now the default and the flag is deprecated
    TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    if (!network) {
        logger->error("Failed to create TensorRT network definition");
        return false;
    }

    // Create ONNX parser
    TRTUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, trtLogger));
    if (!parser) {
        logger->error("Failed to create ONNX parser");
        return false;
    }

    // Parse ONNX model
    std::vector<char> modelData = readFile(onnxFilePath);
    if (modelData.empty()) {
        return false;
    }

    logger->info("Parsing ONNX file: " + onnxFilePath);

    bool parseSuccess = parser->parse(modelData.data(), modelData.size());
    if (!parseSuccess) {
        logger->error("Failed to parse ONNX model");
        return false;
    }

    logger->info("Successfully Parsed ONNX file: " + onnxFilePath);

    // Create config
    TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
        logger->error("Failed to create builder config");
        return false;
    }

    // Set workspace size - smaller workspace to avoid out of memory
    const size_t workspaceSizeMB = 512;
    const size_t workspace_size = static_cast<size_t>(workspaceSizeMB) * (1ULL << 20);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
    logger->info("Setting TensorRT builder workspace size to " + std::to_string(workspaceSizeMB) + "MB");

    // Always enable FP16 mode to match trtexec command
    // config->setFlag(nvinfer1::BuilderFlag::kFP16);
    // config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    // logger->info("Setting TensorRT builder flag to FP16");
    // logger->info("Clearing TensorRT builder flag to TF32");


    // Set builder optimization level to 3 (maximum) to reduce memory usage
    config->setBuilderOptimizationLevel(3);
    logger->info("Setting TensorRT builder optimization level to 3 (maximum)");

    // Create optimization profile with explicit tensor shapes to match trtexec command
    auto profile = builder->createOptimizationProfile();
    bool hasDynamicShapes = false;

    // For each input in the network, check and set dimensions
    for (int i = 0; i < network->getNbInputs(); i++) {
        auto input = network->getInput(i);
        auto dims = input->getDimensions();

        // Check if this input has dynamic dimensions
        bool isDynamicInput = false;
        for (int d = 0; d < dims.nbDims; d++) {
            if (dims.d[d] == -1) {
                isDynamicInput = true;
                hasDynamicShapes = true;
                break;
            }
        }
        // If this is a dynamic input, set dimensions in the profile
        if (isDynamicInput) {
            // Create the min, optimal and max dimensions
            nvinfer1::Dims minDims = dims;
            nvinfer1::Dims optDims = dims;
            nvinfer1::Dims maxDims = dims;

            // Set appropriate values for each dynamic dimension
            for (int d = 0; d < dims.nbDims; d++) {
                if (dims.d[d] == -1) {
                    // model batch size
                    if (d == 0) {
                        minDims.d[d] = 1;
                        optDims.d[d] = batchSize;
                        maxDims.d[d] = batchSize;
                    } else {
                        // model with numSensors
                        minDims.d[d] = 1;
                        optDims.d[d] = numSensors;
                        maxDims.d[d] = numSensors;
                    }
                }
            }

            std::stringstream ss;
            ss << "Input: " << input->getName() << ", Type: ";
            switch (input->getType()) {
                case nvinfer1::DataType::kFLOAT: ss << "FLOAT32"; break;
                case nvinfer1::DataType::kHALF: ss << "FLOAT16"; break;
                case nvinfer1::DataType::kINT8: ss << "INT8"; break;
                case nvinfer1::DataType::kINT32: ss << "INT32"; break;
                case nvinfer1::DataType::kBOOL: ss << "BOOL"; break;
                case nvinfer1::DataType::kUINT8: ss << "UINT8"; break;
                default: ss << "UNKNOWN"; break;
            }
            ss << ", Dims: [";
            for (int d = 0; d < dims.nbDims; d++) {
                ss << dims.d[d];
                if (d < dims.nbDims - 1) ss << ", ";
            }
            ss << "]";
            ss << "  MiDims: [";
            for (int d = 0; d < minDims.nbDims; d++) {
                ss << minDims.d[d];
                if (d < minDims.nbDims - 1) ss << ", ";
            }
            ss << "]";
            ss << "  OptDims: [";
            for (int d = 0; d < optDims.nbDims; d++) {
                ss << optDims.d[d];
                if (d < optDims.nbDims - 1) ss << ", ";
            }
            ss << "]";
            ss << "  MaxDims: [";
            for (int d = 0; d < maxDims.nbDims; d++) {
                ss << maxDims.d[d];
                if (d < maxDims.nbDims - 1) ss << ", ";
            }
            ss << "]";
            logger->info(ss.str());

            // Set dimensions in the profile
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, minDims);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, optDims);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims);
        }
    }
    // Only add the profile if dynamic shapes were detected
    if (hasDynamicShapes) {
        config->addOptimizationProfile(profile);
        logger->info("Added optimization profile to TensorRT builder");
    }

    logger->persistent_info("INFO: Building TensorRT engine from ONNX model " + onnxFilePath);

    // In TensorRT 10.3, use buildSerializedNetwork
    TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine(builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine) {
        logger->error("Failed to build TensorRT engine");
        return false;
    } else {
        logger->info("Success: Built TensorRT Engine");
    }

    // Save engine to file
    std::ofstream engineFile(engineFilePath, std::ios::binary);
    if (!engineFile) {
        logger->error("Failed to open engine file for writing: " + engineFilePath);
        return false;
    }

    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    if (engineFile.fail()) {
        logger->error("Failed to write engine to file");
        return false;
    } else {
        logger->persistent_info("Success: Saved engine to file: " + engineFilePath);
    }

    return true;
}