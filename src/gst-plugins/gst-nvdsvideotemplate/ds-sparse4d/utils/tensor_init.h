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

#ifndef TENSOR_INIT_H
#define TENSOR_INIT_H

#include <Eigen/Dense>
#include "tensor.h"
#include "log_utils.h"
#include "../infer/tensor_rt_engine.h"

/**
 * @brief Creates a tensor with the specified shape, data type, and GPU ID
 *
 * @param tensor_name Name for the tensor
 * @param shape Vector of dimensions for the tensor shape
 * @param data_type Data type for the tensor (defaults to FP32)
 * @param gpu_id GPU ID for memory allocation (-1 for CPU memory, >=0 for GPU)
 * @return Tensor* Newly created tensor with allocated memory
 */
Tensor* createTensor(const std::string& tensor_name,
                    const std::vector<int>& shape,
                    NvDsDataType data_type = NvDsDataType_FP32,
                    int gpu_id = -1);

/**
 * @brief Initialize projection matrix tensor and update the TensorCache
 *
 * @param tensorCache Pointer to TensorCache object
 * @param batch_size Sparse4D model batch size
 * @param num_sensors  Number of sensors [Deepstream batch size]
 * @param calib_matrices Vector of calibration matrices (4x4) to use
 * @param gpu_id GPU ID for memory allocation (-1 for CPU memory, >=0 for GPU)
 */
void initOrgProjectionMatTensor(TensorCache* tensorCache, int batch_size, int num_sensors, const std::vector<Eigen::Matrix<float, 4, 4>>& calib_matrices, int gpu_id);


/**
 * @brief Initializes projection matrix tensor using pre-allocated memory and updates the TensorCache
 *
 * @param tensorCache Pointer to TensorCache object
 * @param batch_size Sparse4D model batch size
 * @param num_sensors  Number of sensors [Deepstream batch size]
 * @param calib_matrices Vector of calibration matrices (4x4) to use
 */
void initProjectionMatTensor(TensorCache* tensorCache, int batch_size, int num_sensors,
                             const std::vector<Eigen::Matrix<float, 4, 4>>& calib_matrices);

/**
 * @brief Initializes image width/height tensor using pre-allocated memory and updates the TensorCache
 * 
 * @param tensorCache Pointer to TensorCache object
 * @param width Width of the image
 * @param height Height of the image
 */
void initImageWHTensor(TensorCache* tensorCache, int width, int height);

/**
 * @brief Initializes cached anchor tensor using pre-allocated memory and updates the TensorCache
 *
 * @param tensorCache Pointer to TensorCache object
 */
void initInputCachedAnchorTensor(TensorCache* tensorCache);

/**
 * @brief Initializes cached feature tensor using pre-allocated memory and updates the TensorCache
 *
 * @param tensorCache Pointer to TensorCache object
 */
void initInputCachedFeatureTensor(TensorCache* tensorCache);

/**
 * @brief Initializes prev_exists tensor using pre-allocated memory and updates the TensorCache
 *
 * @param tensorCache Pointer to TensorCache object
 * @param value Boolean value (0.0 = false, non-zero = true)
 */
void intiPrevExistsTensor(TensorCache* tensorCache, float value);

/**
 * @brief Initializes interval mask tensor using pre-allocated memory and updates the TensorCache
 *
 * @param tensorCache Pointer to TensorCache object
 * @param value Value for the mask (typically 1.0)
 */
void initIntervalMaskTensor(TensorCache* tensorCache, float value);

/**
 * @brief Initializes group origin tensor using pre-allocated memory and updates the TensorCache
 *
 * @param tensorCache Pointer to TensorCache object
 * @param group_origin Vector of group origin values
 */
void initGroupOriginTensor(TensorCache* tensorCache, const std::vector<float>& group_origin);

/**
 * @brief Updates input cached anchor tensor using pre-allocated memory
 *
 * @param tensorCache Pointer to TensorCache object
 * @param anchor Anchor tensor to update
 */
void updateInputCachedAnchorTensor(TensorCache* tensorCache, torch::Tensor anchor);

/**
 * @brief Updates input cached feature tensor using pre-allocated memory
 *
 * @param tensorCache Pointer to TensorCache object
 * @param feature Feature tensor to update
 */
void updateInputCachedFeatureTensor(TensorCache* tensorCache, torch::Tensor feature);

/**
 * @brief Updates projection matrix tensor based on the current camera names in the batch
 *
 * @param tensorCache Pointer to TensorCache object
 * @param cameraMatrices Map of camera names to their projection matrices
 * @param currentBatchCameras Vector of current camera names in the batch
 * @param batch_size Sparse4D model batch size
 * @param num_sensors Maximum number of sensors supported by the model
 */
void updateProjectionMatrixForBatch(TensorCache* tensorCache, 
                                   const std::map<std::string, Eigen::Matrix<float, 4, 4>>& cameraMatrices,
                                   const std::vector<std::string>& currentBatchCameras,
                                   int batch_size,
                                   int num_sensors);

/**
 * @brief Updates the original projection matrix tensor based on the current camera names in the batch
 * 
 * @param tensorCache Pointer to TensorCache object
 * @param cameraMatrices Map of camera names to their projection matrices
 * @param currentBatchCameras Vector of current camera names in the batch
 * @param batch_size Sparse4D model batch size
 * @param num_sensors Maximum number of sensors supported by the model
 */
void updateOrgProjectionMatrixForBatch(TensorCache* tensorCache, 
                                      const std::map<std::string, Eigen::Matrix<float, 4, 4>>& cameraMatrices,
                                      const std::vector<std::string>& currentBatchCameras,
                                      int batch_size, int num_sensors);
#endif /* TENSOR_INIT_H */