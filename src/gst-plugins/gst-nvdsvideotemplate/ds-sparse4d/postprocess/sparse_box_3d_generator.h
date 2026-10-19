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

#ifndef SPARSE_BOX_3D_GENERATOR_H
#define SPARSE_BOX_3D_GENERATOR_H

#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <variant>
#include <iostream>
#include "instance_bank.h"

/**
 * @brief Class for generating 3D bounding box key points based on anchors
 *
 * This class implements the AnchorHandler interface for generating key points
 * from 3D box anchors and providing anchor projection functionality.
 */
class SparseBox3DKeyPointsGenerator : public AnchorHandler, public torch::nn::Module {
public:
    /**
     * @brief Constructor for SparseBox3DKeyPointsGenerator
     *
     * @param embed_dims Dimension of embedding features
     * @param num_learnable_pts Number of learnable key points 
     * @param fix_scale Fixed scale values for key points (array of 3D points)
     */
    SparseBox3DKeyPointsGenerator(
        int embed_dims = 256,
        int num_learnable_pts = 0,
        const std::vector<std::tuple<float, float, float>>& fix_scale = {{0.0, 0.0, 0.0}}
    );

    /**
     * @brief Initialize weights with proper initialization schemes
     */
    void init_weight();

    /**
     * @brief Forward pass to generate key points from anchors
     *
     * @param anchor Anchor tensors [batch_size, num_anchor, dim]
     * @param instance_feature Optional instance features for learnable points
     * @param T_cur2temp_list Optional transformation matrices
     * @param cur_timestamp Optional current timestamp
     * @param temp_timestamps Optional temporal timestamps
     * @return key points tensor or tuple of key points and temporal key points
     */
    std::variant<torch::Tensor, std::tuple<torch::Tensor, std::vector<torch::Tensor>>>
    forward(
        const torch::Tensor& anchor,
        const torch::Tensor& instance_feature = torch::Tensor(),
        const std::vector<torch::Tensor>& T_cur2temp_list = {},
        const torch::Tensor& cur_timestamp = torch::Tensor(),
        const std::vector<torch::Tensor>& temp_timestamps = {}
    );

    /**
     * @brief Project anchors using transformation matrices and time intervals
     *
     * @param anchors Input anchors to project
     * @param transforms Transformation matrices for coordinate transforms
     * @param time_intervals Time intervals for temporal projection
     * @return Vector of projected anchor tensors
     */
    std::vector<torch::Tensor> anchor_projection(
        const torch::Tensor& anchors,
        const std::vector<torch::Tensor>& transforms,
        const std::vector<torch::Tensor>& time_intervals) override;

    /**
     * @brief Static method to project anchors using various parameters
     *
     * @param anchor Input anchor tensor
     * @param T_src2dst_list List of transformation matrices
     * @param src_timestamp Optional source timestamp
     * @param dst_timestamps Optional destination timestamps
     * @param time_intervals Optional time intervals
     * @return Vector of projected anchor tensors
     */
    static std::vector<torch::Tensor> anchor_projection_static(
        const torch::Tensor& anchor,
        const std::vector<torch::Tensor>& T_src2dst_list,
        const torch::Tensor& src_timestamp = torch::Tensor(),
        const std::vector<torch::Tensor>& dst_timestamps = {},
        const std::vector<torch::Tensor>& time_intervals = {}
    );

    /**
     * @brief Calculate Euclidean distance of anchors from origin in 2D
     *
     * @param anchor Input anchor tensor
     * @return Tensor of distances
     */
    static torch::Tensor distance(const torch::Tensor& anchor);

private:
    int embed_dims;
    int num_learnable_pts;
    int num_pts;
    torch::Tensor fix_scale;
    torch::nn::Linear learnable_fc{nullptr};
};

#endif // SPARSE_BOX_3D_GENERATOR_H