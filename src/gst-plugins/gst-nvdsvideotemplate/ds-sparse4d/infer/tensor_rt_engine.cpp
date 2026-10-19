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

#include "tensor_rt_engine.h"


// Logger implementation
void TensorRTEngine::Logger::log(Severity severity, const char* msg) noexcept {
    // Check against configured severity level
    if (static_cast<int>(severity) <= m_Parent->m_LoggerSeverity) {
        const char* severityStr = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: severityStr = "[INTERNAL_ERROR]"; break;
            case Severity::kERROR:          severityStr = "[ERROR]"; break;
            case Severity::kWARNING:        severityStr = "[WARNING]"; break;
            case Severity::kINFO:           severityStr = "[INFO]"; break;
            case Severity::kVERBOSE:        severityStr = "[VERBOSE]"; break;
            default:                        severityStr = "[UNKNOWN]"; break;
        }
        std::cout << "TensorRT " << severityStr << " " << msg << std::endl;
    }
}

// Constructor
TensorRTEngine::TensorRTEngine()
    : m_Engine(nullptr),
      m_Context(nullptr),
      m_Runtime(nullptr),
      m_CudaStream(nullptr),
      m_GpuId(0),
      m_Initialized(false),
      m_LoggerSeverity(2),      // Default to kWARNING level
      m_EnableProfiler(false) {  // Default to disabled
}

// Destructor
TensorRTEngine::~TensorRTEngine() {
    // Free device buffers
    for (void* buffer : m_DeviceBuffers) {
        if (buffer) {
            cudaFree(buffer);
        }
    }

    // Destroy TensorRT objects - using delete instead of destroy()
    if (m_Context) {
        delete m_Context;
        m_Context = nullptr;
    }

    if (m_Engine) {
        delete m_Engine;
        m_Engine = nullptr;
    }

    if (m_Runtime) {
        delete m_Runtime;
        m_Runtime = nullptr;
    }

    // Destroy CUDA stream
    if (m_CudaStream) {
        cudaStreamDestroy(m_CudaStream);
        m_CudaStream = nullptr;
    }
}

// Deserialize engine from file
bool TensorRTEngine::deserializeEngine(const std::string& engineFilePath, TensorCache* tensorCache, int batchSize, int numSensors, PrettyLogger* logger) {
    // Read the engine file
    std::ifstream engineFile(engineFilePath, std::ios::binary);
    if (!engineFile) {
        logger->error("Error opening engine file: " + engineFilePath);
        return false;
    }

    // Set GPU device based on config
    cudaError_t setDeviceError = cudaSetDevice(m_GpuId);
    if (setDeviceError != cudaSuccess) {
        logger->error("Failed to set GPU device " + std::to_string(m_GpuId) + ": " +
                      std::string(cudaGetErrorString(setDeviceError)));
        return false;
    }

    // Get the file size
    engineFile.seekg(0, std::ios::end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);

    // Read the engine into a buffer
    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);

    // Create a runtime to deserialize the engine
    m_Runtime = nvinfer1::createInferRuntime(m_Logger);
    if (!m_Runtime) {
        logger->error("Failed to create TensorRT runtime");
        return false;
    }

    // Deserialize the engine
    m_Engine = m_Runtime->deserializeCudaEngine(engineData.data(), engineSize);

    if (!m_Engine) {
        logger->error("Failed to deserialize CUDA engine");
        return false;
    }

    // Create an execution context
    m_Context = m_Engine->createExecutionContext();
    if (!m_Context) {
        logger->error("Failed to create execution context");
        delete m_Engine;
        m_Engine = nullptr;
        return false;
    }

    // Enable profiling for runtime layer timing
    if (m_EnableProfiler) {
        m_Context->setProfiler(&m_Profiler);
        logger->info("TensorRT profiler enabled for layer timing");
    }

    // Create CUDA stream for async operations
    cudaError_t error = cudaStreamCreate(&m_CudaStream);
    if (error != cudaSuccess) {
        logger->error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(error)));
        return false;
    }

    // Allocate device buffers for input and output tensors
    // In newer TensorRT versions, we need to use the network directly
    int nbIOTensors = m_Engine->getNbIOTensors();
    m_DeviceBuffers.resize(nbIOTensors);

    // Print tensor information and get calculated sizes
    std::vector<size_t> tensorSizes = printTensorInfo(nbIOTensors, tensorCache, batchSize, numSensors, logger);

    // Allocate memory for each tensor
    bool allocationSuccess = true;
    for (int i = 0; i < nbIOTensors; i++) {
        const char* tensorName = m_Engine->getIOTensorName(i);

        // Try to allocate memory with cudaMalloc
        cudaError_t err = cudaMalloc(&m_DeviceBuffers[i], tensorSizes[i]);
        if (err != cudaSuccess) {
            logger->error("Failed to allocate device memory for tensor " + std::to_string(i) +
                         " (" + std::string(tensorName) + "), size: " + std::to_string(tensorSizes[i]) +
                         ": " + std::string(cudaGetErrorString(err)));

            // Clean up previously allocated buffers
            for (int j = 0; j < i; j++) {
                if (m_DeviceBuffers[j]) {
                    cudaFree(m_DeviceBuffers[j]);
                    m_DeviceBuffers[j] = nullptr;
                }
            }

            allocationSuccess = false;
            break;
        }
        // Update tensor buffer with device buffer
        tensorCache->updateTensorBuffer(std::string(tensorName), m_DeviceBuffers[i]);
    }

    if (!allocationSuccess) {
        logger->error("Failed to allocate memory for all tensors");
        return false;
    } else {
        logger->info("Allocated GPU memory for all tensors successfully");
        // logger->info("Device buffers:");
        // for (int i = 0; i < nbIOTensors; i++) {
        //     std::stringstream ss;
        //     ss << "\t" << std::string(m_Engine->getIOTensorName(i)) << ": " << m_DeviceBuffers[i];
        //     logger->info(ss.str());
        // }
    }

    m_Initialized = true;
    logger->info("Deserialized TensorRT Engine Successfully");
    return true;
}

std::vector<size_t> TensorRTEngine::printTensorInfo(int nbIOTensors, TensorCache* tensorCache, int batchSize, int numSensors, PrettyLogger* logger) {
    // Initialize return values
    std::vector<size_t> tensorSizes(nbIOTensors);
    size_t totalMemoryRequired = 0;

    // Log the tensor information table
    std::stringstream table;
    const int col1_width = 9;      // ID - increased for proper padding
    const int col2_width = 34;     // Tensor Name
    const int col3_width = 26;     // Dimensions - increased for longer dimensions with UTF-8 chars
    const int col4_width = 12;     // Type
    const int col5_width = 12;     // Size

    // Calculate total table width
    const int totalTableWidth = col1_width + col2_width + col3_width + col4_width + col5_width + 14; // +14 for borders and padding

    // Create the horizontal line with proper widths
    const std::string horizontalLine = "+" + std::string(col1_width + 2, '-') + "+"
                                         + std::string(col2_width + 2, '-') + "+"
                                         + std::string(col3_width + 2, '-') + "+"
                                         + std::string(col4_width + 2, '-') + "+"
                                         + std::string(col5_width + 2, '-') + "+";

    // Table header with consistent border
    table << "\n" << horizontalLine << "\n";

    // Column headers with proper alignment and padding
    table << "| " << std::setw(col1_width) << std::left << "ID" << " | "
          << std::setw(col2_width) << std::left << "Tensor Name" << " | "
          << std::setw(col3_width) << std::left << "Dimensions" << " | "
          << std::setw(col4_width) << std::left << "Data Type" << " | "
          << std::setw(col5_width) << std::left << "Size" << " |\n";

    table << horizontalLine << "\n";

    // Group tensors by type (inputs vs outputs)
    std::vector<int> inputIndices;
    std::vector<int> outputIndices;

    for (int i = 0; i < nbIOTensors; i++) {
        const char* tensorName = m_Engine->getIOTensorName(i);
        nvinfer1::TensorIOMode ioMode = m_Engine->getTensorIOMode(tensorName);

        if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
            inputIndices.push_back(i);
        } else {
            outputIndices.push_back(i);
        }
    }

    // Function to add a row with color coding based on tensor type
    auto addRow = [&](int i, TensorCache* tensorCache) {
        const char* tensorName = m_Engine->getIOTensorName(i);
        nvinfer1::Dims dims = m_Engine->getTensorShape(tensorName);
        nvinfer1::DataType dtype = m_Engine->getTensorDataType(tensorName);

        // Build dimensions string - replace the Unicode multiplication symbol with regular ASCII for consistency
        std::stringstream dimStr;
        std::vector<int> tensorShape;

        for (int d = 0; d < dims.nbDims; d++) {
            // Handle dynamic dimension (-1) by using a safe default size
            if (dims.d[d] == -1) {
                if (d == 0) {
                    // Batch dimension
                    dimStr << batchSize;
                    tensorShape.push_back(batchSize);
                } else {
                    // Dimension with number of sensors
                    dimStr << numSensors;
                    tensorShape.push_back(numSensors);
                }
            } else {
                // Fixed dimension
                dimStr << dims.d[d];
                tensorShape.push_back(dims.d[d]);
            }
            if (d < dims.nbDims - 1) dimStr << "x";  // Using regular ASCII 'x' instead of Unicode '×'
        }

        // Format the dimensions with specific padding for alignment
        std::string dimString = dimStr.str();
        std::string paddedDim = std::string(2, ' ') + dimString;  // Add 2 leading spaces

        // Calculate needed padding spaces to ensure proper alignment
        int paddingSpaces = std::max(0, col3_width - static_cast<int>(paddedDim.length()));

        // Get data type string
        std::string dtypeStr = getDataTypeString(dtype);
        size_t size = 1;

        // Calculate size and determine type string
        for (int j = 0; j < dims.nbDims; j++) {
            if (dims.d[j] == -1) {
                if (j == 0) size *= batchSize;
                else size *= numSensors;
            } else {
                size *= dims.d[j];
            }
        }

        size *= getElementSize(dtype);

        // Format human-readable size
        std::string sizeStr;
        if (size < 1024) {
            sizeStr = std::to_string(size) + " B";
        } else if (size < 1024 * 1024) {
            sizeStr = std::to_string(size / 1024) + " KB";
        } else {
            double sizeMB = static_cast<double>(size) / (1024 * 1024);
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << sizeMB << " MB";
            sizeStr = ss.str();
        }

        // Add the formatted row with all columns properly aligned
        table << "| " << std::setw(col1_width) << std::left << i << " | "
              << std::setw(col2_width) << std::left << (std::string(tensorName).substr(0, col2_width)) << " | "
              << std::left << paddedDim << std::string(paddingSpaces, ' ') << " | "
              << std::setw(col4_width) << std::left << dtypeStr << " | "
              << std::setw(col5_width) << std::left << sizeStr << " |\n";

        // Store size for memory calculation
        tensorSizes[i] = size;
        totalMemoryRequired += size;

        Tensor* tensor = new Tensor();
        tensor->name = std::string(tensorName);
        tensor->shape = tensorShape;
        tensor->size = size;
        tensor->gpu_id = m_GpuId;
        tensor->frame_num = 0;
        tensor->buffer = nullptr;
        // Set tensor data type based on TensorRT data type
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT:
                tensor->data_type = NvDsDataType_FP32;
                break;
            case nvinfer1::DataType::kHALF:
                tensor->data_type = NvDsDataType_FP16;
                break;
            case nvinfer1::DataType::kINT8:
                tensor->data_type = NvDsDataType_INT8;
                break;
            case nvinfer1::DataType::kINT32:
                tensor->data_type = NvDsDataType_INT32;
                break;
            case nvinfer1::DataType::kBOOL:
                tensor->data_type = NvDsDataType_UINT8;
                break;
            default:
                tensor->data_type = NvDsDataType_FP32;
                break;
        }
        // Add tensor to cache
        tensorCache->addTensor(tensor);
    };

    for (int idx : inputIndices) {
        addRow(idx, tensorCache);
    }

    table << horizontalLine << "\n";
    for (int idx : outputIndices) {
        addRow(idx, tensorCache);
    }

    // Add table footer
    table << horizontalLine << "\n";

    // Add memory summary
    std::string memSizeStr;
    if (totalMemoryRequired < 1024) {
        memSizeStr = std::to_string(totalMemoryRequired) + " B";
    } else if (totalMemoryRequired < 1024 * 1024) {
        memSizeStr = std::to_string(totalMemoryRequired / 1024) + " KB";
    } else {
        double sizeMB = static_cast<double>(totalMemoryRequired) / (1024 * 1024);
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << sizeMB << " MB";
        memSizeStr = ss.str();
    }

    // Create centered memory summary line with proper width
    std::string memSummary = "Total memory required: " + memSizeStr;

    // Center the summary text
    int padding = (totalTableWidth - memSummary.length() - 4) / 2; // -4 for "| " and " |"
    padding = std::max(0, padding);

    // Calculate any extra padding needed for perfect centering
    int extraPadding = totalTableWidth - (2*padding + memSummary.length()) - 2;
    extraPadding = std::max(0, extraPadding);

    // Add the summary row with perfect centering
    table << "| " << std::string(padding, ' ')
          << memSummary
          << std::string(padding + extraPadding, ' ')
          << " |\n";

    // Final border line
    table << "+" << std::string(totalTableWidth, '-') << "+";

    // Log the tensor information table
    logger->persistent_info(table.str());

    // Return calculated tensor sizes
    return tensorSizes;
}

// Run inference with pre-allocated output tensors
bool TensorRTEngine::runInference(int batch_size, int num_sensors, PrettyLogger* logger) {
    if (!m_Initialized || !m_Engine || !m_Context) {
        logger->warning("TensorRT Engine not initialized");
        return false;
    }

    // Get the number of IO tensors and identify input/output tensors
    std::vector<std::string> inputTensorNames = getInputTensorNames();
    std::vector<std::string> outputTensorNames = getOutputTensorNames();

    if (inputTensorNames.empty() || outputTensorNames.empty()) {
        logger->warning("Failed to find input or output tensors");
        return false;
    }

    // Set input shapes for any dynamic input tensors first
    for (size_t i = 0; i < inputTensorNames.size(); i++) {
        const char* tensorName = inputTensorNames[i].c_str();

        nvinfer1::Dims dims = m_Engine->getTensorShape(tensorName);

        // Check if this tensor has any dynamic dimensions (-1)
        bool hasDynamicDims = false;
        for (int d = 0; d < dims.nbDims; d++) {
            if (dims.d[d] == -1) {
                hasDynamicDims = true;
                break;
            }
        }

        if (hasDynamicDims) {
            // Create a new dims object with actual values instead of -1
            nvinfer1::Dims actualDims;
            actualDims.nbDims = dims.nbDims;

            // Set actual dimensions
            for (int d = 0; d < dims.nbDims; d++) {
                if (dims.d[d] == -1) {
                    if (d == 0) {
                        // Batch dimension
                        actualDims.d[d] = batch_size;
                    } else {
                        // Dimension with number of sensors
                        actualDims.d[d] = num_sensors;
                    }
                } else {
                    // Use the original dimension
                    actualDims.d[d] = dims.d[d];
                }
            }
            m_Context->setInputShape(tensorName, actualDims);
        }
    }

    // Set input tensor addresses from existing device buffers
    for (size_t i = 0; i < inputTensorNames.size(); i++) {
        m_Context->setTensorAddress(inputTensorNames[i].c_str(), m_DeviceBuffers[i]);
    }

    // Set output tensor addresses based on device buffers
    for (size_t i = 0; i < outputTensorNames.size(); i++) {
        int idx = i + inputTensorNames.size();
        const std::string& tensorName = outputTensorNames[i];
        m_Context->setTensorAddress(tensorName.c_str(), m_DeviceBuffers[idx]);
    }

    // Execute inference
    bool status = m_Context->enqueueV3(m_CudaStream);
    if (!status) {
        logger->error("Failed to run inference");
        return false;
    }

    // Synchronize the CUDA stream to ensure inference is complete
    cudaError_t syncError = cudaStreamSynchronize(m_CudaStream);
    if (syncError != cudaSuccess) {
        logger->error("Failed to synchronize CUDA stream after inference: " +
                     std::string(cudaGetErrorString(syncError)));
        return false;
    }

    return true;
}