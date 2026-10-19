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

#include "tensor_init.h"

Tensor* createTensor(const std::string& tensor_name,
                    const std::vector<int>& shape,
                    NvDsDataType data_type,
                    int gpu_id)
{
    // Check for empty shape
    if (shape.empty()) {
        std::cerr << "Error in createTensor: Empty shape provided for tensor '" << tensor_name << "'" << std::endl;
        return nullptr;
    }

    // Check for zero dimensions
    for (int dim : shape) {
        if (dim <= 0) {
            std::cerr << "Error in createTensor: Invalid dimension " << dim << " for tensor '" << tensor_name << "'" << std::endl;
            return nullptr;
        }
    }

    Tensor* tensor = new Tensor();
    tensor->name = tensor_name;
    tensor->shape = shape;
    tensor->data_type = data_type;
    tensor->gpu_id = gpu_id;

    // Calculate buffer size based on data type
    size_t element_size;
    switch (data_type) {
    case NvDsDataType_FP32:
        element_size = sizeof(float);
        break;
    case NvDsDataType_FP16:
        element_size = sizeof(short); // half-precision float
        break;
    case NvDsDataType_INT32:
        element_size = sizeof(int32_t);
        break;
    case NvDsDataType_INT8:
        element_size = sizeof(int8_t);
        break;
    case NvDsDataType_UINT8:
        element_size = sizeof(uint8_t);
        break;
    default:
        element_size = sizeof(float); // default to float
        std::cerr << "Warning in createTensor: Unknown data type for tensor '" << tensor_name << "', defaulting to float" << std::endl;
        break;
    }

    // Calculate total buffer size
    tensor->size = element_size;
    for (int dim : shape) {
        tensor->size *= dim;
    }

    // Check for buffer size overflow
    if (tensor->size == 0 || tensor->size > 1024 * 1024 * 1024) { // 1 GB limit as a sanity check
        std::cerr << "Error in createTensor: Invalid buffer size " << tensor->size << " for tensor '"
                  << tensor_name << "'. Check if dimensions are too large." << std::endl;
        delete tensor;
        return nullptr;
    }

    // Allocate memory based on gpu_id
    if (gpu_id < 0) {
        // CPU memory allocation
        try {
            tensor->buffer = malloc(tensor->size);
            if (!tensor->buffer) {
                std::cerr << "Error in createTensor: CPU memory allocation failed for tensor '"
                         << tensor_name << "' with size " << tensor->size << " bytes." << std::endl;
                delete tensor;
                return nullptr;
            }
            // Initialize CPU memory to zero
            std::memset(tensor->buffer, 0, tensor->size);
        } catch (const std::exception& e) {
            std::cerr << "Error in createTensor: CPU memory allocation failed with exception: "
                     << e.what() << std::endl;
            delete tensor;
            return nullptr;
        }
    } else {
        // GPU memory allocation
        cudaError_t cuda_status = cudaMalloc(&tensor->buffer, tensor->size);
        if (cuda_status != cudaSuccess || tensor->buffer == nullptr) {
            std::cerr << "Error in createTensor: CUDA memory allocation failed for tensor '"
                      << tensor_name << "' with size " << tensor->size << " bytes. "
                      << "CUDA error: " << cudaGetErrorString(cuda_status) << std::endl;
            delete tensor;
            return nullptr;
        }

        // Initialize GPU memory to zero
        cuda_status = cudaMemset(tensor->buffer, 0, tensor->size);
        if (cuda_status != cudaSuccess) {
            std::cerr << "Warning in createTensor: Failed to initialize GPU memory for tensor '"
                      << tensor_name << "'. CUDA error: " << cudaGetErrorString(cuda_status) << std::endl;
            cudaFree(tensor->buffer);
            delete tensor;
            return nullptr;
        }
    }

    return tensor;
}

void initOrgProjectionMatTensor(TensorCache* tensorCache, int batch_size, int num_sensors, const std::vector<Eigen::Matrix<float, 4, 4>>& calib_matrices, int gpu_id) {
    // Create a projection matrix tensor and update the TensorCache
    Tensor* tensor = new Tensor();
    tensor->name = "org_projection_mat";
    tensor->data_type = NvDsDataType_FP32;
    tensor->gpu_id = gpu_id;

    // B X N X 4 X 4
    tensor->shape.push_back(batch_size);
    tensor->shape.push_back(num_sensors);
    tensor->shape.push_back(4);
    tensor->shape.push_back(4);
    tensor->size = sizeof(float) * batch_size * num_sensors * 4 * 4;

    // Use the pre-allocated memory buffer
    tensor->buffer = malloc(tensor->size);
    std::memset(tensor->buffer, 0, tensor->size);

    // Create projection matrix data on CPU
    float* host_data = new float[batch_size * num_sensors * 4 * 4];

    // Fill with calibration matrices if available, otherwise use identity matrices
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < num_sensors; i++) {
            // Get calibration matrix for this camera
            Eigen::Matrix<float, 4, 4> calib_mat;
            if (i < (int)calib_matrices.size()) {
                calib_mat = calib_matrices[i];
            } else {
                // Use identity matrix if no calibration matrix available
                calib_mat = Eigen::Matrix<float, 4, 4>::Identity();
                // logger->warning("No calibration matrix available for camera " + std::to_string(i) + ". Using identity matrix.");
            }

            // Copy to host buffer (row-major order)
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    int idx = b * num_sensors * 16 + i * 16 + r * 4 + c;
                    host_data[idx] = calib_mat(r, c);
                }
            }
        }
    }

    // Copy data to tensor buffer
    if (tensor->buffer) {
        std::memcpy(tensor->buffer, host_data, tensor->size);
    }

    // Cleanup host memory
    delete[] host_data;

    tensorCache->addTensor(tensor);
}

void initProjectionMatTensor(TensorCache* tensorCache, int batch_size, int num_sensors, const std::vector<Eigen::Matrix<float, 4, 4>>& calib_matrices) {
    // Use the pre-allocated memory buffer
    void* buffer = tensorCache->getTensorBuffer("projection_mat");
    size_t size = tensorCache->getTensorSize("projection_mat");

    // Create projection matrix data on CPU
    float* host_data = new float[batch_size * num_sensors * 4 * 4];

    // Fill with calibration matrices if available, otherwise use identity matrices
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < num_sensors; i++) {
            // Get calibration matrix for this camera
            Eigen::Matrix<float, 4, 4> calib_mat;
            if (i < (int)calib_matrices.size()) {
                calib_mat = calib_matrices[i];
            } else {
                // Use identity matrix if no calibration matrix available
                calib_mat = Eigen::Matrix<float, 4, 4>::Identity();
                // logger->warning("No calibration matrix available for camera " + std::to_string(i) + ". Using identity matrix.");
            }

            // Copy to host buffer (row-major order)
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    int idx = b * num_sensors * 16 + i * 16 + r * 4 + c;
                    host_data[idx] = calib_mat(r, c);
                }
            }
        }
    }

    // Copy data to GPU (using the pre-allocated buffer)
    if (buffer) {
        cudaMemcpy(buffer, host_data, size, cudaMemcpyHostToDevice);
    }

    // Cleanup host memory
    delete[] host_data;
}

void initImageWHTensor(TensorCache* tensorCache, int width, int height) {
    // Use the pre-allocated memory buffer
    void* buffer = tensorCache->getTensorBuffer("image_wh");
    size_t size = tensorCache->getTensorSize("image_wh");

    // Create width/height data on CPU - use default values if we can't determine from engine
    float* host_data = new float[size / sizeof(float)];

    // Initialize all values to the selected width/height
    for (size_t i = 0; i < size / sizeof(float); i += 2) {
        host_data[i] = static_cast<float>(width);
        host_data[i + 1] = static_cast<float>(height);
    }

    // Copy data to GPU (using the pre-allocated buffer)
    if (buffer) {
        cudaMemcpy(buffer, host_data, size, cudaMemcpyHostToDevice);
    }

    // Cleanup host memory
    delete[] host_data;
}

void initInputCachedAnchorTensor(TensorCache* tensorCache) {
    // Use the pre-allocated memory buffer
    void* buffer = tensorCache->getTensorBuffer("input_cached_anchor");
    size_t size = tensorCache->getTensorSize("input_cached_anchor");

    // Initialize with zeros if we have a valid buffer
    if (buffer) {
        cudaMemset(buffer, 0, size);
    }
}

void initInputCachedFeatureTensor(TensorCache* tensorCache) {
    // Use the pre-allocated memory buffer
    void* buffer = tensorCache->getTensorBuffer("input_cached_feature");
    size_t size = tensorCache->getTensorSize("input_cached_feature");

    // Initialize with zeros if we have a valid buffer
    if (buffer) {
        cudaMemset(buffer, 0, size);
    }
}

void intiPrevExistsTensor(TensorCache* tensorCache, float value) {
    // Use the pre-allocated memory buffer
    void* buffer = tensorCache->getTensorBuffer("prev_exists");
    size_t size = tensorCache->getTensorSize("prev_exists");

    // Create prev_exists data on CPU (0.0 = false, non-zero = true)
    float* host_data = new float[size / sizeof(float)];
    for (size_t i = 0; i < size / sizeof(float); i++) {
        host_data[i] = value;
    }

    // Copy data to GPU (using the pre-allocated buffer)
    if (buffer) {
        cudaMemcpy(buffer, host_data, size, cudaMemcpyHostToDevice);
    }

    // Cleanup host memory
    delete[] host_data;
}

void initIntervalMaskTensor(TensorCache* tensorCache, float value) {
    // Use the pre-allocated memory buffer
    void* buffer = tensorCache->getTensorBuffer("interval_mask");
    size_t size = tensorCache->getTensorSize("interval_mask");

    // Create interval mask data on CPU (0 = false, non-zero = true)
    uint8_t* host_data = new uint8_t[size / sizeof(uint8_t)];
    for (size_t i = 0; i < size / sizeof(uint8_t); i++) {
        host_data[i] = value;
    }

    // Copy data to GPU (using the pre-allocated buffer)
    if (buffer) {
        cudaMemcpy(buffer, host_data, size, cudaMemcpyHostToDevice);
    }

    // Cleanup host memory
    delete[] host_data;
}

void initGroupOriginTensor(TensorCache* tensorCache, const std::vector<float>& group_origin) {
    // Check for valid input data
    if (group_origin.empty()) {
        std::cerr << "Error in initGroupOriginTensor: Empty group_origin vector provided" << std::endl;
        return;
    }

    // Create a tensor to hold the group origin values
    Tensor* tensor = new Tensor();
    tensor->name = "group_origin";
    tensor->data_type = NvDsDataType_FP32;
    tensor->gpu_id = -1; // CPU memory

    // Set shape to match the group_origin vector (typically 2 values)
    tensor->shape.push_back(group_origin.size());
    tensor->size = sizeof(float) * group_origin.size();

    // Allocate memory for tensor
    tensor->buffer = malloc(tensor->size);
    if (!tensor->buffer) {
        std::cerr << "Error in initGroupOriginTensor: Failed to allocate memory for group_origin tensor" << std::endl;
        delete tensor;
        return;
    }

    // Copy group_origin values to tensor buffer
    std::memcpy(tensor->buffer, group_origin.data(), tensor->size);

    // Add tensor to cache
    tensorCache->addTensor(tensor);
}

void updateInputCachedAnchorTensor(TensorCache* tensorCache, torch::Tensor anchor) {
    // Use the pre-allocated memory buffer
    void* buffer = tensorCache->getTensorBuffer("input_cached_anchor");
    size_t size = tensorCache->getTensorSize("input_cached_anchor");

    // Copy data from anchor tensor to cached anchor tensor
    if (buffer) {
        // Check if anchor tensor is on GPU
        if (anchor.is_cuda()) {
            // GPU to GPU copy
            cudaMemcpy(buffer, anchor.data_ptr(), size, cudaMemcpyDeviceToDevice);
        } else {
            // CPU to GPU copy
            cudaMemcpy(buffer, anchor.data_ptr(), size, cudaMemcpyHostToDevice);
        }
    }
}

void updateInputCachedFeatureTensor(TensorCache* tensorCache, torch::Tensor feature) {
    // Use the pre-allocated memory buffer
    void* buffer = tensorCache->getTensorBuffer("input_cached_feature");
    size_t size = tensorCache->getTensorSize("input_cached_feature");

    // Copy data from feature tensor to cached feature tensor
    if (buffer) {
        // Check if feature tensor is on GPU
        if (feature.is_cuda()) {
            // GPU to GPU copy
            cudaMemcpy(buffer, feature.data_ptr(), size, cudaMemcpyDeviceToDevice);
        } else {
            // CPU to GPU copy
            cudaMemcpy(buffer, feature.data_ptr(), size, cudaMemcpyHostToDevice);
        }
    }
}

void updateProjectionMatrixForBatch(TensorCache* tensorCache, 
                                   const std::map<std::string, Eigen::Matrix<float, 4, 4>>& cameraMatrices,
                                   const std::vector<std::string>& currentBatchCameras,
                                   int batch_size,
                                   int num_sensors) {
    // Get pre-allocated buffers for projection matrices
    void* buffer = tensorCache->getTensorBuffer("projection_mat");
    size_t size = tensorCache->getTensorSize("projection_mat");

    int current_num_sensors = currentBatchCameras.size(); // Actual sensors in this batch

    // Create host buffer for the data
    float* host_data = new float[batch_size * num_sensors * 4 * 4];

    // Initialize buffer with zeros
    std::memset(host_data, 0, batch_size * num_sensors * 4 * 4 * sizeof(float));

    // Fill with projection matrices for cameras in the current batch
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < current_num_sensors; i++) {
            const std::string& camera_name = currentBatchCameras[i];

            // Find the projection matrix for this camera
            auto it = cameraMatrices.find(camera_name);
            if (it == cameraMatrices.end()) {
                std::cerr << "Warning: No projection matrix found for camera " << camera_name 
                          << ". Using identity matrix." << std::endl;

                // Use identity matrix
                for (int r = 0; r < 4; r++) {
                    for (int c = 0; c < 4; c++) {
                        int idx = b * num_sensors * 16 + i * 16 + r * 4 + c;
                        host_data[idx] = (r == c) ? 1.0f : 0.0f; // Identity matrix
                    }
                }
            } else {
                // Use the camera's projection matrix
                const auto& proj_mat = it->second;
                for (int r = 0; r < 4; r++) {
                    for (int c = 0; c < 4; c++) {
                        int idx = b * num_sensors * 16 + i * 16 + r * 4 + c;
                        host_data[idx] = proj_mat(r, c);
                    }
                }
            }
        }
    }

    // Copy data to GPU buffer
    if (buffer) {
        cudaError_t cuda_status = cudaMemcpy(buffer, host_data, size, cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            std::cerr << "Error: Failed to copy projection matrix data to GPU. CUDA error: " 
                      << cudaGetErrorString(cuda_status) << std::endl;
        }
    } else {
        std::cerr << "Error: projection_mat buffer is null" << std::endl;
    }
    
    // Cleanup host memory
    delete[] host_data;
}

void updateOrgProjectionMatrixForBatch(TensorCache* tensorCache, 
                                      const std::map<std::string, Eigen::Matrix<float, 4, 4>>& cameraMatrices,
                                      const std::vector<std::string>& currentBatchCameras,
                                      int batch_size,
                                      int num_sensors) {
    // Get pre-allocated buffers for projection matrices
    void* buffer = tensorCache->getTensorBuffer("org_projection_mat");
    size_t size = tensorCache->getTensorSize("org_projection_mat");

    int current_num_sensors = currentBatchCameras.size(); // Actual sensors in this batch   

    // Create host buffer for the data
    float* host_data = new float[batch_size * num_sensors * 4 * 4];

    // Initialize buffer with zeros
    std::memset(host_data, 0, batch_size * num_sensors * 4 * 4 * sizeof(float));

    // Fill with projection matrices for cameras in the current batch
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < current_num_sensors; i++) {
            const std::string& camera_name = currentBatchCameras[i];

            // Find the projection matrix for this camera
            auto it = cameraMatrices.find(camera_name);
            if (it == cameraMatrices.end()) {
                std::cerr << "Warning: No projection matrix found for camera " << camera_name 
                          << ". Using identity matrix." << std::endl;

                // Use identity matrix
                for (int r = 0; r < 4; r++) {
                    for (int c = 0; c < 4; c++) {
                        int idx = b * num_sensors * 16 + i * 16 + r * 4 + c;
                        host_data[idx] = (r == c) ? 1.0f : 0.0f; // Identity matrix
                    }
                }
            } else {
                // Use the camera's projection matrix
                const auto& proj_mat = it->second;
                for (int r = 0; r < 4; r++) {
                    for (int c = 0; c < 4; c++) {
                        int idx = b * num_sensors * 16 + i * 16 + r * 4 + c;
                        host_data[idx] = proj_mat(r, c);
                    }
                }
            }
        }
    }

    // Copy data to CPU buffer (org_projection_mat is in CPU memory)
    if (buffer) {
        std::memcpy(buffer, host_data, size);
    } else {
        std::cerr << "Error: org_projection_mat buffer is null" << std::endl;
    }

    // Cleanup host memory
    delete[] host_data;
}   