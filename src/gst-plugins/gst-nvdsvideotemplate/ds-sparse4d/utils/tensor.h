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

#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "nvdsmeta.h"
#include <torch/torch.h>
#include <unordered_map>

/**
 * @brief Structure to store tensor information
 *
 * This structure encapsulates all the relevant information about a tensor,
 * extracted from the NvDsPreProcessTensorMeta object. It provides methods for
 * converting to PyTorch tensors, memory management, and formatted display.
 */
struct Tensor {
    std::string name;              // Name of the tensor
    void* buffer;                  // Pointer to the tensor data
    size_t size;                   // Size of the buffer in bytes
    std::vector<int> shape;        // Shape of the tensor (dimensions)
    NvDsDataType data_type;        // Data type of the tensor
    int gpu_id;                    // GPU ID where tensor is allocated (-1 for CPU)
    int frame_num;                 // Frame number associated with this tensor

    /**
     * @brief Default constructor initializing members to safe defaults
     */
    Tensor() : buffer(nullptr), size(0), shape(), data_type(NvDsDataType_FP32), gpu_id(-1), frame_num(0) {}

    /**
     * @brief Convert Tensor to PyTorch tensor
     *
     * This method converts the Tensor object to a PyTorch tensor.
     * The resulting PyTorch tensor shares memory with the original buffer,
     * so no additional memory allocation occurs.
     *
     * @return torch::Tensor PyTorch tensor created from this Tensor, or empty tensor if buffer is invalid
     */
    inline torch::Tensor to_torch_tensor() const {
        if (!buffer) {
            // Return an empty tensor if buffer is invalid
            return torch::Tensor();
        }

        std::vector<int64_t> torch_shape;
        torch_shape.reserve(shape.size());
        for (auto dim : shape) {
            torch_shape.push_back(static_cast<int64_t>(dim));
        }

        // Create torch tensor options with appropriate device
        auto options = torch::TensorOptions().device(gpu_id >= 0 ?
            torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU));

        // Create torch tensor based on data type
        switch (data_type) {
            case NvDsDataType_FP32: {
                float* data_ptr = static_cast<float*>(buffer);
                return torch::from_blob(data_ptr, torch_shape, torch::kFloat32).to(options.device());
            }
            case NvDsDataType_INT32: {
                int32_t* data_ptr = static_cast<int32_t*>(buffer);
                return torch::from_blob(data_ptr, torch_shape, torch::kInt32).to(options.device());
            }
            case NvDsDataType_INT8: {
                int8_t* data_ptr = static_cast<int8_t*>(buffer);
                return torch::from_blob(data_ptr, torch_shape, torch::kInt8).to(options.device());
            }
            case NvDsDataType_UINT8: {
                uint8_t* data_ptr = static_cast<uint8_t*>(buffer);
                return torch::from_blob(data_ptr, torch_shape, torch::kUInt8).to(options.device());
            }
            case NvDsDataType_INT64: {
                int64_t* data_ptr = static_cast<int64_t*>(buffer);
                return torch::from_blob(data_ptr, torch_shape, torch::kInt64).to(options.device());
            }
            case NvDsDataType_UINT32: {
                uint32_t* data_ptr = static_cast<uint32_t*>(buffer);
                return torch::from_blob(data_ptr, torch_shape, torch::kInt32).to(options.device());
            }
            case NvDsDataType_UINT64: {
                uint64_t* data_ptr = static_cast<uint64_t*>(buffer);
                return torch::from_blob(data_ptr, torch_shape, torch::kInt64).to(options.device());
            }
            case NvDsDataType_FP16: {
                uint16_t* data_ptr = static_cast<uint16_t*>(buffer);
                return torch::from_blob(data_ptr, torch_shape, torch::kHalf).to(options.device());
            }
            default: {
                // For unsupported types, return empty tensor
                std::cerr << "Unsupported data type for conversion to torch::Tensor" << std::endl;
                return torch::Tensor();
            }
        }
    }

    /**
     * @brief Release GPU memory associated with this tensor
     *
     * This method safely deallocates any GPU memory allocated for the tensor
     * and resets the tensor to its initial state. It handles the case where
     * the buffer is already nullptr.
     *
     * @return True if memory was deallocated, false if no memory was allocated
     */
    inline bool release() {
        bool memory_freed = false;

        // Check if buffer exists
        if (buffer) {
            // Determine if memory is on GPU or CPU based on gpu_id
            if (gpu_id >= 0) {
                // GPU memory - use cudaFree
                cudaError_t err = cudaFree(buffer);
                memory_freed = (err == cudaSuccess);
                if (err != cudaSuccess) {
                    std::cerr << "Error freeing GPU memory: " << cudaGetErrorString(err) << std::endl;
                }
            } else {
                // CPU memory - use free
                free(buffer);
                memory_freed = true;
            }
            buffer = nullptr;
        }

        // Reset all members to initial state
        size = 0;
        shape.clear();
        data_type = NvDsDataType_FP32;
        name.clear();
        gpu_id = -1;
        frame_num = 0;

        return memory_freed;
    }

    /**
     * @brief Destructor to automatically release resources
     *
     * Ensures that GPU memory is properly freed when the tensor is destroyed.
     */
    inline ~Tensor() noexcept {
        release();
    }

    /**
     * @brief Display metadata information about this tensor in a formatted table
     */
    inline void display(bool print_data = false) const {
        bool is_gpu_tensor = (gpu_id >= 0);
        size_t total_elements = 1;
        size_t max_elements = 100;
        int precision = 4;
        const int lineWidth = 100; // Total width inside the table
        const std::string headerBorder = "+" + std::string(lineWidth+1, '-') + "+";

        std::cout << "\n" << headerBorder << std::endl;

        // Create header with tensor name
        std::string tensorName = !name.empty() ? name : "Unnamed";
        std::string headerText = "Tensor Information - " + tensorName;

        // Center the header text
        int padding = (lineWidth - headerText.length()) / 2;
        std::cout << "| " << std::string(padding, ' ') << headerText
                << std::string(lineWidth - padding - headerText.length(), ' ') << "|" << std::endl;

        std::cout << headerBorder << std::endl;

        // Fixed width for label and value columns to ensure proper alignment
        const int labelWidth = 16;
        const int valueWidth = lineWidth - labelWidth - 3; // -3 for "| " and " |"

        // Display shape information
        std::string shapeStr = "[";
        if (shape.empty()) {
            if (shape.size() > 0) {
                shapeStr += std::to_string(shape[shape.size() - 1]);
            } else {
                shapeStr += "Unknown";
            }
        } else {
            for (size_t i = 0; i < shape.size(); i++) {
                shapeStr += std::to_string(shape[i]);
                if (i < shape.size() - 1) shapeStr += ", ";
            }
        }
        shapeStr += "]";

        std::string deviceStr = (is_gpu_tensor) ? "GPU [" + std::to_string(gpu_id) + "]" : "CPU";
        std::string typeStr;
        switch (data_type) {
            case NvDsDataType_FP32: typeStr = "Float32"; total_elements = size / sizeof(float); break;
            case NvDsDataType_FP16: typeStr = "Float16"; total_elements = size / sizeof(float); break;
            case NvDsDataType_INT8: typeStr = "Int8"; total_elements = size; break;
            case NvDsDataType_INT32: typeStr = "Int32"; total_elements = size / sizeof(int32_t); break;
            case NvDsDataType_UINT8: typeStr = "UInt8"; total_elements = size; break;
            case NvDsDataType_UINT32: typeStr = "UInt32"; total_elements = size / sizeof(uint32_t); break;
            case NvDsDataType_INT64: typeStr = "Int64"; total_elements = size / sizeof(int64_t); break;
            case NvDsDataType_UINT64: typeStr = "UInt64"; total_elements = size / sizeof(uint64_t); break;
            default: typeStr = "Unknown"; break;
        }
        size_t elements_to_display = std::min(total_elements, max_elements);

        // Display all information with consistent formatting
        std::cout << "| Shape           | " << std::left << std::setw(valueWidth) << shapeStr << " |" << std::endl;
        std::cout << "| Device          | " << std::left << std::setw(valueWidth) << deviceStr << " |" << std::endl;
        std::cout << "| Buffer Size     | " << std::left << std::setw(valueWidth) << std::to_string(size) + " bytes" << " |" << std::endl;
        std::cout << "| Elements        | " << std::left << std::setw(valueWidth) << std::to_string(total_elements) << " |" << std::endl;
        std::cout << "| Data Type       | " << std::left << std::setw(valueWidth) << typeStr << " |" << std::endl;
        if (buffer) {
            std::cout << "| Buffer Pointer  | " << std::left << std::setw(valueWidth) << buffer << " |" << std::endl;
        } else {
            std::cout << "| Buffer Pointer  | " << std::left << std::setw(valueWidth) << "NULL" << " |" << std::endl;
        }

        if (frame_num > 0) {
            std::cout << "| Frame Number    | " << std::left << std::setw(valueWidth) << frame_num << " |" << std::endl;
        }

        if (print_data) {
            // Check if buffer is null before trying to print data
            if (!buffer) {
                std::cout << "|" << std::string(lineWidth+1, '-') << "|" << std::endl;
                std::cout << "| No data to display - buffer is null" << std::string(lineWidth - 34, ' ') << "|" << std::endl;
                std::cout << headerBorder << std::endl;
                return;
            }

            // Divider before data section
            std::cout << "|" << std::string(lineWidth+1, '-') << "|" << std::endl;
            void* host_buffer = nullptr;
            if (is_gpu_tensor) {
                std::cout << "| Values [GPU --> CPU]:" << std::string(lineWidth - 21, ' ') << "|" << std::endl;
                // Allocate CPU memory
                host_buffer = malloc(size);
                if (!host_buffer) {
                    std::cerr << "| ERROR: Failed to allocate CPU memory for tensor display"
                            << std::string(lineWidth - 51, ' ') << "|" << std::endl;
                    std::cout << headerBorder << std::endl;
                    return;
                }
                // Copy data from GPU to CPU
                cudaError_t err = cudaMemcpy(host_buffer, buffer, size, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    std::cerr << "| ERROR: Failed to copy tensor data from GPU to CPU: " << cudaGetErrorString(err)
                            << std::string(lineWidth - 59 - strlen(cudaGetErrorString(err)), ' ') << "|" << std::endl;
                    free(host_buffer);
                    std::cout << headerBorder << std::endl;
                    return;
                }
            } else {
                // Data is already on CPU
                std::cout << "| Values [CPU]:" << std::string(lineWidth - 13, ' ') << "|" << std::endl;
                host_buffer = buffer;
            }

            // Set floating point precision for display
            std::cout << std::fixed << std::setprecision(precision);

            // Print data type
            std::string typeStr;
            switch (data_type) {
                case NvDsDataType_FP32: typeStr = "Float32"; break;
                case NvDsDataType_FP16: typeStr = "Float16"; break;
                case NvDsDataType_INT8: typeStr = "Int8"; break;
                case NvDsDataType_INT32: typeStr = "Int32"; break;
                case NvDsDataType_UINT8: typeStr = "UInt8"; break;
                case NvDsDataType_UINT32: typeStr = "UInt32"; break;
                case NvDsDataType_INT64: typeStr = "Int64"; break;
                case NvDsDataType_UINT64: typeStr = "UInt64"; break;
                default: typeStr = "Unknown"; break;
            }

            // Add one blank line before values
            std::cout << "|" << std::string(lineWidth+1, ' ') << "|" << std::endl;

            const int indent = 2; // Indent from the left border
            const int valueWidth = lineWidth - 3; // Maximum width for values per line
            std::stringstream line;
            int line_chars = 0;

            switch (data_type) {
                case NvDsDataType_FP32: {
                    const float* data = static_cast<const float*>(host_buffer);
                    for (size_t i = 0; i < elements_to_display; i++) {
                        std::stringstream value_stream;
                        value_stream << std::fixed << std::setprecision(precision) << data[i];
                        std::string value = value_stream.str();

                        // If this value would exceed the line width, print the current line and start a new one
                        if (line_chars > 0 && line_chars + value.length() + 2 > valueWidth) { // +2 for ", "
                            std::cout << "| " << std::string(indent, ' ') << line.str() << std::string(valueWidth - line_chars, ' ') << " |" << std::endl;
                            line.str("");
                            line_chars = 0;
                        }

                        // Add the value to the current line
                        if (line_chars > 0) {
                            line << ", ";
                            line_chars += 2;
                        }
                        line << value;
                        line_chars += value.length();
                    }
                    break;
                }
                case NvDsDataType_INT32: {
                    const int32_t* data = static_cast<const int32_t*>(host_buffer);
                    for (size_t i = 0; i < elements_to_display; i++) {
                        std::string value = std::to_string(data[i]);

                        // If this value would exceed the line width, print the current line and start a new one
                        if (line_chars > 0 && line_chars + value.length() + 2 > valueWidth) { // +2 for ", "
                            std::cout << "| " << std::string(indent, ' ') << line.str() << std::string(valueWidth - line_chars, ' ') << " |" << std::endl;
                            line.str("");
                            line_chars = 0;
                        }

                        // Add the value to the current line
                        if (line_chars > 0) {
                            line << ", ";
                            line_chars += 2;
                        }
                        line << value;
                        line_chars += value.length();
                    }
                    break;
                }
                case NvDsDataType_UINT32: {
                    const uint32_t* data = static_cast<const uint32_t*>(host_buffer);
                    for (size_t i = 0; i < elements_to_display; i++) {
                        std::string value = std::to_string(data[i]);

                        // If this value would exceed the line width, print the current line and start a new one
                        if (line_chars > 0 && line_chars + value.length() + 2 > valueWidth) { // +2 for ", "
                            std::cout << "| " << std::string(indent, ' ') << line.str() << std::string(valueWidth - line_chars, ' ') << " |" << std::endl;
                            line.str("");
                            line_chars = 0;
                        }

                        // Add the value to the current line
                        if (line_chars > 0) {
                            line << ", ";
                            line_chars += 2;
                        }
                        line << value;
                        line_chars += value.length();
                    }
                    break;
                }
                case NvDsDataType_INT8: {
                    const int8_t* data = static_cast<const int8_t*>(host_buffer);
                    for (size_t i = 0; i < elements_to_display; i++) {
                        std::string value = std::to_string(static_cast<int>(data[i]));

                        // If this value would exceed the line width, print the current line and start a new one
                        if (line_chars > 0 && line_chars + value.length() + 2 > valueWidth) { // +2 for ", "
                            std::cout << "| " << std::string(indent, ' ') << line.str() << std::string(valueWidth - line_chars, ' ') << " |" << std::endl;
                            line.str("");
                            line_chars = 0;
                        }

                        // Add the value to the current line
                        if (line_chars > 0) {
                            line << ", ";
                            line_chars += 2;
                        }
                        line << value;
                        line_chars += value.length();
                    }
                    break;
                }
                case NvDsDataType_UINT8: {
                    const uint8_t* data = static_cast<const uint8_t*>(host_buffer);
                    for (size_t i = 0; i < elements_to_display; i++) {
                        std::string value = std::to_string(static_cast<int>(data[i]));

                        // If this value would exceed the line width, print the current line and start a new one
                        if (line_chars > 0 && line_chars + value.length() + 2 > valueWidth) { // +2 for ", "
                            std::cout << "| " << std::string(indent, ' ') << line.str() << std::string(valueWidth - line_chars, ' ') << " |" << std::endl;
                            line.str("");
                            line_chars = 0;
                        }

                        // Add the value to the current line
                        if (line_chars > 0) {
                            line << ", ";
                            line_chars += 2;
                        }
                        line << value;
                        line_chars += value.length();
                    }
                    break;
                }
                case NvDsDataType_FP16: {
                    // FP16 display requires conversion to float
                    std::cout << "| " << std::string(indent, ' ') << "FP16 data display not fully implemented"
                            << std::string(valueWidth - 37, ' ') << "|" << std::endl;
                    break;
                }
                case NvDsDataType_INT64: {
                    const int64_t* data = static_cast<const int64_t*>(host_buffer);
                    for (size_t i = 0; i < elements_to_display; i++) {
                        std::string value = std::to_string(data[i]);

                        // If this value would exceed the line width, print the current line and start a new one
                        if (line_chars > 0 && line_chars + value.length() + 2 > valueWidth) { // +2 for ", "
                            std::cout << "| " << std::string(indent, ' ') << line.str() << std::string(valueWidth - line_chars, ' ') << " |" << std::endl;
                            line.str("");
                            line_chars = 0;
                        }

                        // Add the value to the current line
                        if (line_chars > 0) {
                            line << ", ";
                            line_chars += 2;
                        }
                        line << value;
                        line_chars += value.length();
                    }
                    break;
                }
                case NvDsDataType_UINT64: {
                    const uint64_t* data = static_cast<const uint64_t*>(host_buffer);
                    for (size_t i = 0; i < elements_to_display; i++) {
                        std::string value = std::to_string(data[i]);

                        // If this value would exceed the line width, print the current line and start a new one
                        if (line_chars > 0 && line_chars + value.length() + 2 > valueWidth) { // +2 for ", "
                            std::cout << "| " << std::string(indent, ' ') << line.str() << std::string(valueWidth - line_chars, ' ') << " |" << std::endl;
                            line.str("");
                            line_chars = 0;
                        }

                        // Add the value to the current line
                        if (line_chars > 0) {
                            line << ", ";
                            line_chars += 2;
                        }
                        line << value;
                        line_chars += value.length();
                    }
                    break;
                }
                default:
                    std::cout << "| " << std::string(indent, ' ') << "Unsupported data type for display"
                            << std::string(valueWidth - 32, ' ') << "|" << std::endl;
            }

            // Print the last line if there's anything left
            if (line_chars > 0) {
                std::cout << "| " << std::string(indent, ' ') << line.str() << std::string(valueWidth - line_chars, ' ') << " |" << std::endl;
            }

            // If there are more elements than we displayed, indicate that
            if (elements_to_display < total_elements) {
                std::cout << "|" << std::string(lineWidth+1, ' ') << "|" << std::endl;
                std::cout << "| " << std::string(indent, ' ') << "..." << std::string(valueWidth - 3, ' ') << " |" << std::endl;
            }

            // Add one blank line after values
            std::cout << "|" << std::string(lineWidth+1, ' ') << "|" << std::endl;

            // Close the table
            std::cout << headerBorder << std::endl;

            // Free CPU memory if we allocated it
            if (is_gpu_tensor && host_buffer) {
                free(host_buffer);
            }
        }

        std::cout << headerBorder << std::endl;
    }
};

/**
 * @brief Class to manage multiple tensors with string-based lookup
 *
 * This class provides a central repository for caching tensors,
 * allowing easy retrieval by name and automatic memory management.
 * All tensors added to the cache are owned by the cache and will be
 * automatically released when the cache is destroyed.
 */
class TensorCache {
private:
    std::unordered_map<std::string, Tensor*> m_tensors;  // Map of tensor name to tensor pointer

public:
    /**
     * @brief Default constructor
     */
    TensorCache() {}

    /**
     * @brief Destructor that automatically releases all cached tensors
     *
     * Ensures that all memory allocated for tensors is properly freed
     * when the cache is destroyed.
     */
    ~TensorCache() {
        releaseAll();
    }

    /**
     * @brief Add a tensor to the cache
     *
     * @param tensor Pointer to tensor object (ownership transferred to cache)
     * @return True if tensor was added, false if tensor is nullptr
     * @note The cache takes ownership of the tensor and will delete it when appropriate
     */
    bool addTensor(Tensor* tensor) {
        if (tensor) {
            m_tensors[tensor->name] = tensor;
            return true;
        }
        return false;
    }

    /**
     * @brief Get a tensor by name
     *
     * @param name Name of the tensor to retrieve
     * @return Pointer to the tensor, or nullptr if not found
     * @note The returned tensor is still owned by the cache
     */
    Tensor* getTensor(const std::string& name) const {
        auto it = m_tensors.find(name);
        if (it != m_tensors.end()) {
            return it->second;
        }
        return nullptr;
    }

    /**
     * @brief Get a tensor's buffer by name
     *
     * @param name Name of the tensor whose buffer to retrieve
     * @return void* Pointer to the tensor buffer, or nullptr if tensor not found or has no buffer
     */
    void* getTensorBuffer(const std::string& name) const {
        auto it = m_tensors.find(name);
        if (it != m_tensors.end() && it->second) {
            return it->second->buffer;
        }
        return nullptr;
    }

    /**
     * @brief Get a tensor's size by name
     *
     * @param name Name of the tensor whose size to retrieve
     * @return size_t Size of the tensor buffer in bytes, or 0 if tensor not found
     */
    size_t getTensorSize(const std::string& name) const {
        auto it = m_tensors.find(name);
        if (it != m_tensors.end() && it->second) {
            return it->second->size;
        }
        return 0;
    }

    /**
     * @brief Get a tensor's shape by name
     *
     * @param name Name of the tensor whose shape to retrieve
     * @return std::vector<int> Shape of the tensor (dimensions), or empty vector if tensor not found
     */
    std::vector<int> getTensorShape(const std::string& name) const {
        auto it = m_tensors.find(name);
        if (it != m_tensors.end() && it->second) {
            return it->second->shape;
        }
        return std::vector<int>();
    }

    /**
     * @brief Get a tensor's gpu_id by name
     *
     * @param name Name of the tensor whose gpu_id to retrieve
     * @return int GPU ID of the tensor, or -1 if tensor not found or is on CPU
     */
    int getTensorGpuId(const std::string& name) const {
        auto it = m_tensors.find(name);
        if (it != m_tensors.end() && it->second) {
            return it->second->gpu_id;
        }
        return -1;
    }

    /**
     * @brief Get a tensor's frame_num by name
     *
     * @param name Name of the tensor whose frame_num to retrieve
     * @return int Frame number of the tensor, or -1 if tensor not found
     */
    int getTensorFrameNum(const std::string& name) const {
        auto it = m_tensors.find(name);
        if (it != m_tensors.end() && it->second) {
            return it->second->frame_num;
        }
        return -1;
    }

    /**
     * @brief Get a tensor directly as PyTorch tensor by name
     *
     * @param name Name of the tensor to retrieve
     * @return torch::Tensor PyTorch tensor created from the cached tensor, or empty tensor if not found
     * @note The returned PyTorch tensor shares memory with the original tensor buffer
     */
    torch::Tensor getTorchTensor(const std::string& name) const {
        Tensor* tensor = getTensor(name);
        if (tensor && tensor->buffer) {
            return tensor->to_torch_tensor();
        }
        return torch::Tensor();  // Return empty tensor if not found
    }

    /**
     * @brief Update only a tensor's buffer pointer, keeping all other properties the same
     *
     * @param name Name of the tensor to update
     * @param newBuffer Pointer to the new buffer (must be GPU memory with same size/layout)
     * @return True if tensor was found and updated, false otherwise
     */
    bool updateTensorBuffer(const std::string& name, void* newBuffer) {
        Tensor* tensor = getTensor(name);
        if (!tensor) {
            return false;  // Tensor not found
        }

        // Only update the buffer pointer, keep all other properties the same
        tensor->buffer = newBuffer;

        return true;
    }

    /**
     * @brief Clone a tensor to CPU memory
     *
     * Creates a new Tensor object with a CPU buffer and copies data from the existing tensor.
     * Works regardless of whether the source tensor is on CPU or GPU.
     *
     * @param name Name of the tensor to clone
     * @return Pointer to new tensor on CPU, or nullptr if source tensor not found
     *         Caller is responsible for deleting the returned tensor
     */
    Tensor* cloneTensorCPU(const std::string& name) {
        // Get the source tensor
        Tensor* srcTensor = getTensor(name);
        if (!srcTensor || !srcTensor->buffer || srcTensor->size == 0) {
            return nullptr;  // Source tensor not found or invalid
        }

        std::string cpu_tensor_name = srcTensor->name + "_cpu";
        Tensor* cpuTensor = nullptr;

        // Check if the CPU tensor already exists
        if (hasTensor(cpu_tensor_name)) {
            // If it exists, use the already allocated CPU tensor
            cpuTensor = getTensor(cpu_tensor_name);
        } else{
            // Create a new tensor with the same properties
            cpuTensor = new Tensor();
            cpuTensor->name = cpu_tensor_name;
            cpuTensor->shape = srcTensor->shape;
            cpuTensor->data_type = srcTensor->data_type;
            cpuTensor->size = srcTensor->size;
            cpuTensor->gpu_id = -1;

            // Allocate CPU memory
            cpuTensor->buffer = malloc(srcTensor->size);
            if (!cpuTensor->buffer) {
                delete cpuTensor;
                std::cerr << "Failed to allocate CPU memory for tensor clone" << std::endl;
                return nullptr;
            }

            // Add the new tensor to the cache
            m_tensors[cpu_tensor_name] = cpuTensor;
        }

        // Update frame number
        cpuTensor->frame_num = srcTensor->frame_num;

        // Copy data from source tensor to CPU tensor
        if (srcTensor->gpu_id >= 0) {
            // Source is on GPU, need GPU to CPU copy
            cudaError_t copyError = cudaMemcpy(
                cpuTensor->buffer,        // Destination: CPU buffer
                srcTensor->buffer,        // Source: GPU buffer
                srcTensor->size,          // Size to copy
                cudaMemcpyDeviceToHost    // GPU to CPU copy
            );

            if (copyError != cudaSuccess) {
                std::cerr << "Failed to copy tensor from GPU to CPU: "
                          << cudaGetErrorString(copyError) << std::endl;
                free(cpuTensor->buffer);
                delete cpuTensor;
                return nullptr;
            }
        } else {
            // Source is already on CPU, do a CPU to CPU copy
            memcpy(cpuTensor->buffer, srcTensor->buffer, srcTensor->size);
        }

        return cpuTensor;
    }

    /**
     * @brief Clone a tensor to CPU memory and convert to PyTorch tensor
     *
     * Creates a CPU copy of the specified tensor and converts it to a PyTorch tensor.
     * Works regardless of whether the source tensor is on CPU or GPU.
     * The function handles memory management internally, freeing the cloned tensor
     * after creating the PyTorch tensor.
     * 
     * @param name Name of the tensor to clone
     * @return torch::Tensor PyTorch tensor on CPU, or empty tensor if source tensor not found
     */
    inline torch::Tensor cloneTorchTensorCPU(const std::string& name) {
        // Get a CPU clone of the tensor
        Tensor* cpuTensor = cloneTensorCPU(name);
        if (!cpuTensor) {
            return torch::Tensor();  // Return empty tensor if clone failed
        }

        // Convert to PyTorch tensor
        return cpuTensor->to_torch_tensor();
    }

    /**
     * @brief Check if a tensor with the given name exists
     *
     * @param name Name of the tensor to check
     * @return True if tensor exists, false otherwise
     */
    bool hasTensor(const std::string& name) const {
        return m_tensors.find(name) != m_tensors.end();
    }

    /**
     * @brief Remove a tensor from the cache and release its resources
     *
     * @param name Name of the tensor to remove
     * @return True if tensor was found and removed, false otherwise
     */
    bool removeTensor(const std::string& name) {
        auto it = m_tensors.find(name);
        if (it != m_tensors.end()) {
            if (it->second) {
                // Use release() to properly free memory based on whether it's CPU or GPU
                it->second->release();
                delete it->second;  // Delete the tensor object
            }
            m_tensors.erase(it);
            return true;
        }
        return false;
    }

    /**
     * @brief Release all tensors and clear the cache
     */
    void releaseAll() {
        for (auto& pair : m_tensors) {
            // Skip img tensor
            if (pair.first == "img") {
                continue;
            }

            if (pair.second) {
                // Use release() to properly free memory based on whether it's CPU or GPU
                pair.second->release();
                delete pair.second;  // Delete the tensor object
            }
        }
        m_tensors.clear();
    }

    /**
     * @brief Get the number of tensors in the cache
     *
     * @return Number of tensors
     */
    size_t size() const {
        return m_tensors.size();
    }

    /**
     * @brief Get a list of all tensor names in the cache
     *
     * @return Vector of tensor names
     */
    std::vector<std::string> getTensorNames() const {
        std::vector<std::string> names;
        names.reserve(m_tensors.size());
        for (const auto& pair : m_tensors) {
            names.push_back(pair.first);
        }
        return names;
    }

    /**
     * @brief Get all tensors in the cache
     *
     * @return Reference to the map containing all tensor name-pointer pairs
     */
    const std::unordered_map<std::string, Tensor*>& getTensors() const {
        return m_tensors;
    }

    /**
     * @brief Display information about all cached tensors
     *
     * @param print_data Whether to print the tensor data values
     */
    void display(bool print_data = false) const {
        for (const auto& pair : m_tensors) {
            if (pair.second) {
                pair.second->display(print_data);
            } else {
                std::cout << "  [NULL TENSOR]" << std::endl;
            }
        }
    }

    /**
     * @brief Display information about given tensor
     *
     * @param name Name of the tensor to display
     * @param print_data Whether to print the tensor data values
     */
    void displayTensor(const std::string& name, bool print_data = false) const {
        auto it = m_tensors.find(name);
        if (it != m_tensors.end() && it->second) {
            it->second->display(print_data);
        } else {
            std::cout << "  [NULL TENSOR]" << std::endl;
        }
    }

};

/**
 * @brief Provide a string representation of the sizes of a torch tensor
 *
 * Converts the dimensions of a PyTorch tensor to a human-readable string format.
 * For example, a tensor with dimensions [3, 224, 224] would return "[3x224x224]".
 *
 * @param tensor PyTorch tensor whose dimensions to convert
 * @return String representation of tensor dimensions in format [dim1xdim2x...]
 */
inline std::string torch_tensor_sizes_to_string(const torch::Tensor& tensor) {
    if (!tensor.defined()) {
        return "[undefined]";
    }

    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < tensor.sizes().size(); ++i) {
        ss << tensor.sizes()[i];
        if (i < tensor.sizes().size() - 1) {
            ss << "x";
        }
    }
    ss << "]";
    return ss.str();
}

/**
 * @brief Convert torch tensor to string representation of its values
 *
 * This function creates a human-readable string representation of a tensor's values.
 * It handles different data types and allows limiting the number of values displayed.
 *
 * @param tensor PyTorch tensor to convert to string
 * @param max_values Maximum number of values to include in the string, or -1 for all values
 * @return String representation of tensor values in format [val1, val2, ...]
 */
inline std::string torch_tensor_to_string(const torch::Tensor& tensor, int max_values = 10) {
    // Check if tensor is valid
    if (!tensor.defined()) {
        return "[undefined tensor]";
    }

    // Create a copy to avoid modifying the original tensor
    auto t = tensor.clone().contiguous().cpu();

    std::stringstream ss;
    ss << "[";

    // Determine how many values to print
    int64_t num_to_print;
    bool show_ellipsis;

    if (max_values == -1) {
        // Print all elements when max_values is -1
        num_to_print = t.numel();
        show_ellipsis = false;
    } else {
        // Print limited number of elements
        num_to_print = std::min(static_cast<int64_t>(max_values), t.numel());
        show_ellipsis = num_to_print < t.numel();
    }

    // Return early for empty tensors
    if (t.numel() == 0) {
        ss << "]";
        return ss.str();
    }

    // Handle different tensor types
    if (t.scalar_type() == torch::ScalarType::Float) {
        auto a = t.data_ptr<float>();
        for (int64_t i = 0; i < num_to_print; i++) {
            ss << a[i];
            if (i < num_to_print - 1) {
                ss << ", ";
            }
        }
    } else if (t.scalar_type() == torch::ScalarType::Int) {
        auto a = t.data_ptr<int32_t>();
        for (int64_t i = 0; i < num_to_print; i++) {
            ss << a[i];
            if (i < num_to_print - 1) {
                ss << ", ";
            }
        }
    } else if (t.scalar_type() == torch::ScalarType::Long) {
        auto a = t.data_ptr<int64_t>();
        for (int64_t i = 0; i < num_to_print; i++) {
            ss << a[i];
            if (i < num_to_print - 1) {
                ss << ", ";
            }
        }
    } else {
        // Fallback for other types
        ss << "tensor(" << t.sizes() << ")";
        show_ellipsis = false;
    }

    // Add ellipsis if tensor has more elements than we're showing
    if (show_ellipsis) {
        ss << ", ...";
    }

    ss << "]";
    return ss.str();
}

/**
 * @brief Helper function to process tensor data based on its type
 *
 * This function is used by read_tensor_from_file to handle different data types.
 * It allocates memory, reads data from file, and transfers it to GPU memory.
 *
 * @param file Open file stream to read data from
 * @param result Tensor to populate
 * @param dtype Data type string from file
 * @param num_elements Total number of elements in tensor
 * @param shape_int64 Shape of tensor as int64_t vector
 * @param file_path Path to file (for logging)
 * @param gpu_id GPU ID where tensor is allocated
 */
inline void processDataType(
    std::ifstream& file, 
    Tensor* result, 
    const std::string& dtype, 
    int64_t num_elements,
    const std::vector<int64_t>& shape_int64,
    const std::string& file_path,
    int gpu_id) {

    cudaError_t cuda_status;

    // Process float32 data
    if (dtype.find("float32") != std::string::npos) {
        result->data_type = NvDsDataType_FP32;
        size_t element_size = sizeof(float);

        // Allocate memory for reading data on CPU
        std::vector<float> cpu_data(num_elements);
        file.read(reinterpret_cast<char*>(cpu_data.data()), num_elements * element_size);

        // Allocate GPU memory
        result->size = num_elements * element_size;
        cuda_status = cudaMalloc(&result->buffer, result->size);
        if (cuda_status != cudaSuccess) {
            std::cerr << "Error allocating GPU memory: "
                      << cudaGetErrorString(cuda_status) << std::endl;
            return;
        }

        // Copy data from CPU to GPU
        cuda_status = cudaMemcpy(result->buffer, cpu_data.data(), result->size, cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            std::cerr << "Error copying data to GPU: "
                      << cudaGetErrorString(cuda_status) << std::endl;
            cudaFree(result->buffer);
            result->buffer = nullptr;
            return;
        }
    }
}

/**
 * @brief Read a tensor from a binary file created by dump_tensor_to_file
 *
 * This function reads a tensor from a binary file that was saved using the
 * dump_tensor_to_file function in Python. It handles the file format that includes
 * tensor shape and data type information. The data is loaded into GPU memory.
 *
 * Supported data types:
 * - float32 (converted to NvDsDataType_FP32)
 * - float64 (converted to NvDsDataType_FP32)
 * - float16/half (converted to NvDsDataType_FP16)
 * - int32 (converted to NvDsDataType_INT32)
 * - int64 (converted to NvDsDataType_INT64)
 * - int8 (converted to NvDsDataType_INT8)
 * - uint8 (converted to NvDsDataType_UINT8)
 *
 * @param file_path Path to the binary file containing the tensor data
 * @param gpu_id GPU ID where the tensor should be allocated (default 0)
 * @return Tensor* Pointer to a heap-allocated custom Tensor struct with data on GPU
 *         The caller is responsible for freeing this memory
 */
inline Tensor* read_tensor_from_file(const std::string& file_path, int gpu_id = 0) {
    // Initialize result tensor
    Tensor* result = new Tensor();
    result->name = "tensor_from_file";
    result->gpu_id = gpu_id;

    // Open file
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << file_path << std::endl;
        return result;  // Return empty tensor with valid pointer but no data
    }

    try {
        // Read tensor metadata from file header

        int64_t num_dims;
        file.read(reinterpret_cast<char*>(&num_dims), sizeof(int64_t));

        // Read shape dimensions
        std::vector<int64_t> shape_int64(num_dims);
        file.read(reinterpret_cast<char*>(shape_int64.data()), num_dims * sizeof(int64_t));

        // Convert int64_t shape to int shape for our Tensor struct
        result->shape.resize(num_dims);
        for (int64_t i = 0; i < num_dims; ++i) {
            result->shape[i] = static_cast<int>(shape_int64[i]);
        }

        // Read dtype string length
        int64_t dtype_str_len;
        file.read(reinterpret_cast<char*>(&dtype_str_len), sizeof(int64_t));

        // Read dtype string
        std::vector<char> dtype_str(dtype_str_len + 1, 0);  // +1 for null terminator
        file.read(dtype_str.data(), dtype_str_len);
        std::string dtype(dtype_str.data());

        // Calculate total number of elements
        int64_t num_elements = 1;
        for (int64_t dim : shape_int64) {
            num_elements *= dim;
        }

        // Set CUDA device
        cudaError_t cuda_status = cudaSetDevice(gpu_id);
        if (cuda_status != cudaSuccess) {
            std::cerr << "Error setting CUDA device " << gpu_id << ": "
                      << cudaGetErrorString(cuda_status) << std::endl;
            return result;  // Return empty tensor
        }

        // Process different data types
        processDataType(file, result, dtype, num_elements, shape_int64, file_path, gpu_id);

        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading tensor from file: " << e.what() << std::endl;
        if (result->buffer) {
            cudaFree(result->buffer);
            result->buffer = nullptr;
        }
        return result;  // Return empty tensor
    }
}

/**
 * @brief Display a PyTorch tensor in a formatted table
 *
 * This function displays a PyTorch tensor in the same formatted table style as the
 * Tensor::display method. It shows tensor properties and optionally the tensor values.
 *
 * @param tensor The PyTorch tensor to display
 * @param name Optional name for the tensor (defaults to "Tensor")
 * @param print_data Whether to print the tensor data values (defaults to false)
 * @param max_elements Maximum number of elements to display (defaults to 100)
 * @param precision Number of decimal places for floating point values (defaults to 4)
 */
inline void torch_tensor_display(const torch::Tensor& tensor,
                               const std::string& name = "Tensor",
                               bool print_data = false,
                               size_t max_elements = 100,
                               int precision = 4) {
    if (!tensor.defined()) {
        std::cout << "Undefined tensor" << std::endl;
        return;
    }

    // Table formatting constants
    const int lineWidth = 100;
    const std::string headerBorder = "+" + std::string(lineWidth+1, '-') + "+";
    const int labelWidth = 16;
    const int valueWidth = lineWidth - labelWidth - 3; // -3 for "| " and " |"
    const int indent = 2;

    std::cout << "\n" << headerBorder << std::endl;

    // Create header with tensor name
    std::string headerText = "Tensor Information - " + name;
    int padding = (lineWidth - headerText.length()) / 2;
    std::cout << "| " << std::string(padding, ' ') << headerText
            << std::string(lineWidth - padding - headerText.length(), ' ') << "|" << std::endl;

    std::cout << headerBorder << std::endl;

    // Get tensor properties
    auto sizes = tensor.sizes();
    auto dtype = tensor.scalar_type();
    auto device = tensor.device();
    size_t numel = tensor.numel();
    size_t elements_to_display = std::min(numel, max_elements);

    // Format shape as string
    std::string shapeStr = "[";
    for (size_t i = 0; i < sizes.size(); i++) {
        shapeStr += std::to_string(sizes[i]);
        if (i < sizes.size() - 1) shapeStr += ", ";
    }
    shapeStr += "]";

    // Convert dtype to string
    std::string dtypeStr;
    switch (dtype) {
        case torch::kFloat32: dtypeStr = "Float32"; break;
        case torch::kFloat16: dtypeStr = "Float16"; break;
        case torch::kFloat64: dtypeStr = "Float64"; break;
        case torch::kInt32: dtypeStr = "Int32"; break;
        case torch::kInt64: dtypeStr = "Int64"; break;
        case torch::kInt16: dtypeStr = "Int16"; break;
        case torch::kInt8: dtypeStr = "Int8"; break;
        case torch::kUInt8: dtypeStr = "UInt8"; break;
        case torch::kBool: dtypeStr = "Bool"; break;
        default: dtypeStr = "Unknown"; break;
    }

    // Format device information
    std::string deviceStr;
    if (device.is_cuda()) {
        deviceStr = "GPU [" + std::to_string(device.index()) + "]";
    } else {
        deviceStr = "CPU";
    }

    // Calculate memory size (approximate)
    size_t element_size = 0;
    switch (dtype) {
        case torch::kFloat32: element_size = 4; break;
        case torch::kFloat16: element_size = 2; break;
        case torch::kFloat64: element_size = 8; break;
        case torch::kInt32: element_size = 4; break;
        case torch::kInt64: element_size = 8; break;
        case torch::kInt16: element_size = 2; break;
        case torch::kInt8: element_size = 1; break;
        case torch::kUInt8: element_size = 1; break;
        case torch::kBool: element_size = 1; break;
        default: element_size = 4; break;
    }
    size_t memory_size = numel * element_size;

    // Display tensor properties
    std::cout << "| Shape           | " << std::left << std::setw(valueWidth) << shapeStr << " |" << std::endl;
    std::cout << "| Device          | " << std::left << std::setw(valueWidth) << deviceStr << " |" << std::endl;
    std::cout << "| Memory Size     | " << std::left << std::setw(valueWidth) << std::to_string(memory_size) + " bytes" << " |" << std::endl;
    std::cout << "| Elements        | " << std::left << std::setw(valueWidth) << std::to_string(numel) << " |" << std::endl;
    std::cout << "| Data Type       | " << std::left << std::setw(valueWidth) << dtypeStr << " |" << std::endl;
    std::cout << "| Requires Grad   | " << std::left << std::setw(valueWidth) << (tensor.requires_grad() ? "True" : "False") << " |" << std::endl;

    // Print tensor data values if requested
    if (print_data && numel > 0) {
        // Ensure tensor is on CPU for display
        torch::Tensor cpu_tensor = tensor.device().is_cuda() ? tensor.to(torch::kCPU) : tensor;

        // Create contiguous version for easier access
        cpu_tensor = cpu_tensor.contiguous();

        // Divider before data section
        std::cout << "|" << std::string(lineWidth+1, '-') << "|" << std::endl;

        if (tensor.device().is_cuda()) {
            std::cout << "| Values [GPU --> CPU]:" << std::string(lineWidth - 21, ' ') << "|" << std::endl;
        } else {
            std::cout << "| Values [CPU]:" << std::string(lineWidth - 13, ' ') << "|" << std::endl;
        }

        // Add one blank line before values
        std::cout << "|" << std::string(lineWidth+1, ' ') << "|" << std::endl;

        // Set floating point precision
        std::cout << std::fixed << std::setprecision(precision);

        // Prepare all values as strings first
        std::vector<std::string> value_strings;
        value_strings.reserve(elements_to_display);

        // Convert tensor data to strings based on type
        if (dtype == torch::kFloat32) {
            auto data_ptr = cpu_tensor.data_ptr<float>();
            for (size_t i = 0; i < elements_to_display; i++) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(precision) << data_ptr[i];
                value_strings.push_back(ss.str());
            }
        } else if (dtype == torch::kInt32) {
            auto data_ptr = cpu_tensor.data_ptr<int32_t>();
            for (size_t i = 0; i < elements_to_display; i++) {
                value_strings.push_back(std::to_string(data_ptr[i]));
            }
        } else if (dtype == torch::kInt64) {
            auto data_ptr = cpu_tensor.data_ptr<int64_t>();
            for (size_t i = 0; i < elements_to_display; i++) {
                value_strings.push_back(std::to_string(data_ptr[i]));
            }
        } else if (dtype == torch::kUInt8 || dtype == torch::kInt8) {
            auto data_ptr = cpu_tensor.data_ptr<uint8_t>();
            for (size_t i = 0; i < elements_to_display; i++) {
                value_strings.push_back(std::to_string(static_cast<int>(data_ptr[i])));
            }
        } else if (dtype == torch::kBool) {
            auto data_ptr = cpu_tensor.data_ptr<bool>();
            for (size_t i = 0; i < elements_to_display; i++) {
                value_strings.push_back(data_ptr[i] ? "true" : "false");
            }
        } else if (dtype == torch::kFloat16) {
            std::cout << "| " << std::string(indent, ' ') << "Float16 data display not fully implemented"
                    << std::string(valueWidth - 37, ' ') << "|" << std::endl;
            std::cout << "|" << std::string(lineWidth+1, ' ') << "|" << std::endl;
            std::cout << headerBorder << std::endl;
            return;
        } else {
            std::cout << "| " << std::string(indent, ' ') << "Unsupported data type for display"
                    << std::string(valueWidth - 32, ' ') << "|" << std::endl;
            std::cout << "|" << std::string(lineWidth+1, ' ') << "|" << std::endl;
            std::cout << headerBorder << std::endl;
            return;
        }

        // Format values into lines with proper wrapping
        const int max_line_width = lineWidth - 2 * indent - 4; // -4 for "| " and " |"
        std::string current_line;
        int current_line_width = 0;

        for (size_t i = 0; i < value_strings.size(); i++) {
            std::string value = value_strings[i];
            std::string entry = value;

            // Add comma if not the last element
            if (i < value_strings.size() - 1) {
                entry += ", ";
            }

            int entry_width = entry.length();

            // Check if adding this entry would exceed the line width
            if (current_line_width > 0 && current_line_width + entry_width > max_line_width) {
                // Print current line and start a new one
                std::cout << "| " << std::string(indent, ' ') << std::left
                         << std::setw(lineWidth - 3) << current_line << " |" << std::endl;
                current_line = entry;
                current_line_width = entry_width;
            } else {
                // Add to current line
                current_line += entry;
                current_line_width += entry_width;
            }
        }

        // Print the last line if anything left
        if (current_line_width > 0) {
            std::cout << "| " << std::string(indent, ' ') << std::left
                     << std::setw(lineWidth - 3) << current_line << " |" << std::endl;
        }

        // If there are more elements than displayed, add ellipsis
        if (elements_to_display < numel) {
            std::cout << "|" << std::string(lineWidth+1, ' ') << "|" << std::endl;
            std::cout << "| " << std::string(indent, ' ') << "..."
                    << std::string(lineWidth - indent - 5, ' ') << " |" << std::endl;
        }

        // Add one blank line after values
        std::cout << "|" << std::string(lineWidth+1, ' ') << "|" << std::endl;
    }

    // Close the table
    std::cout << headerBorder << std::endl;
}

#endif /* TENSOR_H */