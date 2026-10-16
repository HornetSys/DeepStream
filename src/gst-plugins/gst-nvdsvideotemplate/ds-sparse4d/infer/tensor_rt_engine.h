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

#ifndef TENSOR_RT_ENGINE_H
#define TENSOR_RT_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "../utils/log_utils.h"
#include "../utils/tensor.h"
#include "../utils/config_properties.h"

/**
 * @brief TensorRT inference engine class
 *
 * This class handles the deserialization of TensorRT engines and running inference.
 */
class TensorRTEngine {
public:
    /**
     * @brief Constructor
     */
    TensorRTEngine();

    /**
     * @brief Destructor - frees allocated resources
     */
    ~TensorRTEngine();

    /**
     * @brief Check if the engine is initialized
     *
     * @return true if engine is loaded and ready, false otherwise
     */
    bool isInitialized() const { return m_Initialized; }

    /**
     * @brief Configure the TensorRT engine with properties from config
     *
     * @param config Configuration properties containing GPU ID, logger severity, and profiler settings
     */
    void configure(const ConfigProperties& config) {
        m_GpuId = config.gpu_id;
        m_LoggerSeverity = config.trt_logger_severity;
        m_EnableProfiler = config.enable_profiler;
    }

    /**
     * @brief Get the GPU ID used for inference
     *
     * @return The GPU ID currently in use
     */
    int getGpuId() const { return m_GpuId; }

    /**
     * @brief Get the TensorRT engine object
     *
     * @return Pointer to ICudaEngine
     */
    nvinfer1::ICudaEngine* getEngine() const { return m_Engine; }

    /**
     * @brief Get CUDA stream
     *
     * @return CUDA stream used for inference
     */
    cudaStream_t getCudaStream() const { return m_CudaStream; }

    /**
     * @brief Get binding index for a tensor by name
     *
     * @param name Tensor name
     * @return Binding index, or -1 if tensor not found
     */
    int getTensorBindingIndex(const std::string& name) const {
        if (!m_Initialized || !m_Engine) {
            return -1;
        }

        for (int i = 0; i < static_cast<int>(m_Engine->getNbIOTensors()); i++) {
            if (name == m_Engine->getIOTensorName(i)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * @brief Get binding dimensions for a tensor by index
     *
     * @param index Binding index
     * @return Tensor dimensions
     */
    nvinfer1::Dims getBindingDimensions(int index) const {
        if (m_Engine && index >= 0 && index < static_cast<int>(m_Engine->getNbIOTensors())) {
            return m_Engine->getTensorShape(m_Engine->getIOTensorName(index));
        }
        nvinfer1::Dims emptyDims;
        emptyDims.nbDims = 0;
        return emptyDims;
    }

    /**
     * @brief Get device buffer for a tensor by name
     *
     * @param name Tensor name
     * @return Pointer to device buffer, or nullptr if tensor not found
     */
    void* getDeviceBuffer(const std::string& name) const {
        int idx = getTensorBindingIndex(name);
        if (idx >= 0 && static_cast<size_t>(idx) < m_DeviceBuffers.size()) {
            return m_DeviceBuffers[idx];
        }
        return nullptr;
    }

    /**
     * @brief Get all input tensor names
     *
     * @return Vector of input tensor names
     */
    std::vector<std::string> getInputTensorNames() const {
        if (!m_Initialized || !m_Engine) {
            return std::vector<std::string>();
        }

        // Return cached tensor names if available
        if (!m_InputTensorNames.empty()) {
            return m_InputTensorNames;
        }

        // Otherwise, build the list
        std::vector<std::string> inputTensorNames;
        int nbIOTensors = m_Engine->getNbIOTensors();
        
        for (int i = 0; i < nbIOTensors; i++) {
            const char* tensorName = m_Engine->getIOTensorName(i);
            nvinfer1::TensorIOMode tensorMode = m_Engine->getTensorIOMode(tensorName);
            
            if (tensorMode == nvinfer1::TensorIOMode::kINPUT) {
                inputTensorNames.push_back(tensorName);
            }
        }

        // Cache the results
        const_cast<TensorRTEngine*>(this)->m_InputTensorNames = inputTensorNames;
        return inputTensorNames;
    }

    /**
     * @brief Get all output tensor names
     *
     * @return Vector of output tensor names
     */
    std::vector<std::string> getOutputTensorNames() const {
        if (!m_Initialized || !m_Engine) {
            return std::vector<std::string>();
        }

        // Return cached tensor names if available
        if (!m_OutputTensorNames.empty()) {
            return m_OutputTensorNames;
        }

        // Otherwise, build the list
        std::vector<std::string> outputTensorNames;
        int nbIOTensors = m_Engine->getNbIOTensors();

        for (int i = 0; i < nbIOTensors; i++) {
            const char* tensorName = m_Engine->getIOTensorName(i);
            nvinfer1::TensorIOMode tensorMode = m_Engine->getTensorIOMode(tensorName);

            if (tensorMode == nvinfer1::TensorIOMode::kOUTPUT) {
                outputTensorNames.push_back(tensorName);
            }
        }

        // Cache the results
        const_cast<TensorRTEngine*>(this)->m_OutputTensorNames = outputTensorNames;
        return outputTensorNames;
    }

    /**
     * @brief Updates the binding pointer for a specific tensor
     * 
     * Instead of copying tensor data, this method directly updates the
     * binding pointer to use an external buffer. The external buffer must 
     * be in GPU memory with the same size as the original tensor.
     *
     * @param tensorName The name of the tensor to update
     * @param newBuffer Pointer to the new buffer in GPU memory
     * @return true if successful, false otherwise
     */
    bool updateBindingPointer(const std::string& tensorName, void* newBuffer) {
        if (!m_Initialized || !m_Engine || !m_Context) {
            return false;
        }

        int bindingIdx = getTensorBindingIndex(tensorName);
        if (bindingIdx < 0) {
            return false;
        }

        // // Free old buffer memory to prevent memory leaks
        // void* oldBuffer = m_DeviceBuffers[bindingIdx];
        // if (oldBuffer != nullptr && oldBuffer != newBuffer) {
        //     cudaError_t err = cudaFree(oldBuffer);
        //     if (err != cudaSuccess) {
        //         // Log error but continue with update
        //         std::cerr << "Warning: Failed to free old buffer for tensor '" 
        //                   << tensorName << "': " << cudaGetErrorString(err) << std::endl;
        //     }
        // }

        m_DeviceBuffers[bindingIdx] = newBuffer;
        m_Context->setTensorAddress(tensorName.c_str(), newBuffer);
        return true;
    }

    /**
     * @brief Deserialize a TensorRT engine from a file
     *
     * @param engineFilePath Path to the serialized engine file
     * @param tensorCache Tensor cache to store tensors
     * @param batchSize Sparse4D model batch size
     * @param numSensors Number of sensors [Deepstream batch size
     * @param logger Optional logger for detailed status messages
     * @return true if deserialization succeeded, false otherwise
     */
    bool deserializeEngine(const std::string& engineFilePath, TensorCache* tensorCache, int batchSize, int numSensors, PrettyLogger* logger = nullptr);

    /**
     * @brief Run inference using pre-allocated output tensors
     *
     * @param batch_size Sparse4D model batch size
     * @param num_sensors Number of cameras (for dynamic tensor dimensions)
     * @param logger Optional logger for detailed status messages
     * @return true if inference succeeded, false otherwise
     */
    bool runInference(int batch_size, int num_sensors, PrettyLogger* logger = nullptr);

private:
    // TensorRT logger
    class Logger : public nvinfer1::ILogger {
    public:
        Logger(TensorRTEngine* parent) : m_Parent(parent) {}
        void log(Severity severity, const char* msg) noexcept override;
    private:
        TensorRTEngine* m_Parent;
    };

    // Simple profiler for runtime information
    class SimpleProfiler : public nvinfer1::IProfiler {
    public:
        void reportLayerTime(const char* layerName, float timeMs) noexcept override {
            std::cout << "TensorRT [PROFILE] Layer: " << layerName 
                      << " - Time: " << std::fixed << std::setprecision(3) 
                      << timeMs << " ms" << std::endl;
        }
    };

    // TensorRT engine and execution context
    nvinfer1::ICudaEngine* m_Engine;
    nvinfer1::IExecutionContext* m_Context;

    // TensorRT runtime objects
    nvinfer1::IRuntime* m_Runtime;
    Logger m_Logger{this};  // Initialize with parent pointer
    SimpleProfiler m_Profiler;

    // Device buffers for input/output tensors
    std::vector<void*> m_DeviceBuffers;

    // CUDA stream for asynchronous operations
    cudaStream_t m_CudaStream;

    // GPU ID to use for inference
    int m_GpuId;

    // Flag to track initialization status
    bool m_Initialized;

    // Cached input and output tensor names
    std::vector<std::string> m_InputTensorNames;
    std::vector<std::string> m_OutputTensorNames;

    // Configuration properties for logger and profiler
    int m_LoggerSeverity;      // Logger severity level
    bool m_EnableProfiler;     // Whether to enable profiler

    /**
     * @brief Helper function to get element size in bytes based on data type
     * 
     * @param dataType TensorRT data type
     * @return Size of one element of the specified type in bytes
     */
    size_t getElementSize(nvinfer1::DataType dataType) const {
        switch (dataType) {
            case nvinfer1::DataType::kFLOAT:
                return 4; // 4 bytes for float
            case nvinfer1::DataType::kHALF:
                return 2; // 2 bytes for half
            case nvinfer1::DataType::kINT8:
                return 1; // 1 byte for int8
            case nvinfer1::DataType::kINT32:
                return 4; // 4 bytes for int32
            case nvinfer1::DataType::kBOOL:
                return 1; // 1 byte for bool
            default:
                return 4; // Default to 4 bytes
        }
    }

    /** 
     * @brief Get the DataType String from TensorRT data type
     * 
     * @param dataType  TensorRT data type
     * @return String of the data type
     */
    std::string getDataTypeString(nvinfer1::DataType dataType) const {
        switch (dataType) {
            case nvinfer1::DataType::kFLOAT: return "FP32";
            case nvinfer1::DataType::kHALF: return "FP16";
            case nvinfer1::DataType::kINT8: return "INT8";
            case nvinfer1::DataType::kINT32: return "INT32";
            case nvinfer1::DataType::kBOOL: return "BOOL";
            default: return "UNKNOWN";
        }
    }

    /**
     * @brief Prints detailed information about all tensors in the TensorRT engine
     *
     * @param nbIOTensors Number of input/output tensors in the engine
     * @param tensorCache Tensor cache to store tensors
     * @param batchSize Sparse4D model batch size
     * @param numSensors Number of sensors [Deepstream batch size]
     * @param logger Logger to output the formatted table
     * @return Vector of calculated tensor sizes
     */
    std::vector<size_t> printTensorInfo(int nbIOTensors, TensorCache* tensorCache, int batchSize, int numSensors, PrettyLogger* logger);

};

#endif // TENSOR_RT_ENGINE_H