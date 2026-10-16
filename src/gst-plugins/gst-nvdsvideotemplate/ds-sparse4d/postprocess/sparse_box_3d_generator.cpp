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

#include "sparse_box_3d_generator.h"

/**
 * @brief Constants for 3D bounding box anchor components
 */
// Anchor components indices
constexpr int X = 0;        // Center X coordinate
constexpr int Y = 1;        // Center Y coordinate
constexpr int Z = 2;        // Center Z coordinate
constexpr int W = 3;        // Width
constexpr int L = 4;        // Length
constexpr int H = 5;        // Height
constexpr int COS_YAW = 6;  // Cosine of yaw angle
constexpr int SIN_YAW = 7;  // Sine of yaw angle
constexpr int VX = 8;       // Velocity X component


SparseBox3DKeyPointsGenerator::SparseBox3DKeyPointsGenerator(
    int embed_dims,
    int num_learnable_pts,
    const std::vector<std::tuple<float, float, float>>& fix_scale
) : embed_dims(embed_dims), num_learnable_pts(num_learnable_pts) {
    // Convert fix_scale vector to tensor representation
    std::vector<float> flattened_scale;
    for (const auto& point : fix_scale) {
        flattened_scale.push_back(std::get<0>(point));
        flattened_scale.push_back(std::get<1>(point));
        flattened_scale.push_back(std::get<2>(point));
    }

    this->fix_scale = torch::from_blob(
        flattened_scale.data(),
        {static_cast<int64_t>(fix_scale.size()), 3},
        torch::kFloat32
    ).clone();

    this->num_pts = fix_scale.size() + num_learnable_pts;

    // Create learnable linear layer if needed
    if (num_learnable_pts > 0) {
        learnable_fc = register_module(
            "learnable_fc",
            torch::nn::Linear(embed_dims, num_learnable_pts * 3)
        );
    }
}

/**
 * @brief Initialize weights with proper initialization schemes
 */
void SparseBox3DKeyPointsGenerator::init_weight() {
    if (num_learnable_pts > 0) {
        torch::nn::init::xavier_uniform_(learnable_fc->weight);
        torch::nn::init::constant_(learnable_fc->bias, 0.0);
    }
}


std::variant<torch::Tensor, std::tuple<torch::Tensor, std::vector<torch::Tensor>>>
SparseBox3DKeyPointsGenerator::forward(
    const torch::Tensor& anchor,
    const torch::Tensor& instance_feature,
    const std::vector<torch::Tensor>& T_cur2temp_list,
    const torch::Tensor& cur_timestamp,
    const std::vector<torch::Tensor>& temp_timestamps
) {
    auto bs = anchor.size(0);
    auto num_anchor = anchor.size(1);

    // Create scale tensor from fixed scale
    auto scale = fix_scale.unsqueeze(0).unsqueeze(0).repeat({bs, num_anchor, 1, 1});

    // Add learnable scale if available
    if (num_learnable_pts > 0 && instance_feature.defined()) {
        auto learnable_scale = learnable_fc(instance_feature)
            .reshape({bs, num_anchor, num_learnable_pts, 3})
            .sigmoid() - 0.5;

        // Ensure same dtype before concatenation
        if (learnable_scale.dtype() != scale.dtype()) {
            learnable_scale = learnable_scale.to(scale.dtype());
        }

        scale = torch::cat({scale, learnable_scale}, -2);
    }

    // Calculate key points from scale and anchor
    auto boxes_wlh = anchor.index({"...", torch::indexing::Slice(),
                                 torch::indexing::Slice({W, L, H})});
    auto key_points = scale * boxes_wlh.exp().unsqueeze(-2);

    // Create rotation matrix
    auto rotation_mat = torch::zeros({bs, num_anchor, 3, 3}, anchor.options());
    rotation_mat.index_put_({"...", 0, 0}, anchor.index({"...", COS_YAW}));
    rotation_mat.index_put_({"...", 0, 1}, -anchor.index({"...", SIN_YAW}));
    rotation_mat.index_put_({"...", 1, 0}, anchor.index({"...", SIN_YAW}));
    rotation_mat.index_put_({"...", 1, 1}, anchor.index({"...", COS_YAW}));
    rotation_mat.index_put_({"...", 2, 2}, 1);

    // Apply rotation to key points
    key_points = torch::matmul(
        rotation_mat.unsqueeze(2),
        key_points.unsqueeze(-1)
    ).squeeze(-1);

    // Ensure same dtype before addition
    auto center_xyz = anchor.index({"...", torch::indexing::Slice(),
                                  torch::indexing::Slice({X, Y, Z})})
                            .unsqueeze(-2);

    if (key_points.dtype() != center_xyz.dtype()) {
        key_points = key_points.to(center_xyz.dtype());
    }

    // Add center offset
    key_points = key_points + center_xyz;

    // Return if temporal info not available
    if (!cur_timestamp.defined() || temp_timestamps.empty() || T_cur2temp_list.empty()) {
        return key_points;
    }

    // Process temporal key points
    std::vector<torch::Tensor> temp_key_points_list;
    auto velocity = anchor.index({"...", torch::indexing::Slice(VX, torch::indexing::None)});

    for (size_t i = 0; i < temp_timestamps.size(); i++) {
        auto time_interval = cur_timestamp - temp_timestamps[i];
        auto translation = velocity *
            time_interval.to(velocity.dtype()).unsqueeze(1).unsqueeze(2);

        // Ensure same dtype before subtraction
        if (translation.dtype() != key_points.dtype()) {
            translation = translation.to(key_points.dtype());
        }

        auto temp_key_points = key_points - translation.unsqueeze(-2);
        auto T_cur2temp = T_cur2temp_list[i].to(key_points.dtype());

        // Ensure consistent dtypes for concatenation
        auto ones_tensor = torch::ones_like(temp_key_points.index({"...", torch::indexing::Slice(0, 1)}));
        if (ones_tensor.dtype() != temp_key_points.dtype()) {
            ones_tensor = ones_tensor.to(temp_key_points.dtype());
        }

        temp_key_points = torch::matmul(
            T_cur2temp.unsqueeze(1).unsqueeze(1).index({"...", torch::indexing::Slice(0, 3)}),
            torch::cat({temp_key_points, ones_tensor}, -1).unsqueeze(-1)
        ).squeeze(-1);

        temp_key_points_list.push_back(temp_key_points);
    }

    return std::make_tuple(key_points, temp_key_points_list);
}


std::vector<torch::Tensor> SparseBox3DKeyPointsGenerator::anchor_projection(
    const torch::Tensor& anchors,
    const std::vector<torch::Tensor>& transforms,
    const std::vector<torch::Tensor>& time_intervals) {

    return anchor_projection_static(
        anchors,
        transforms,
        torch::Tensor(),
        std::vector<torch::Tensor>(),
        time_intervals);
}


std::vector<torch::Tensor> SparseBox3DKeyPointsGenerator::anchor_projection_static(
    const torch::Tensor& anchor,
    const std::vector<torch::Tensor>& T_src2dst_list,
    const torch::Tensor& src_timestamp,
    const std::vector<torch::Tensor>& dst_timestamps,
    const std::vector<torch::Tensor>& time_intervals
) {
    std::vector<torch::Tensor> dst_anchors;

    for (size_t i = 0; i < T_src2dst_list.size(); i++) {
        auto vel = anchor.index({"...", torch::indexing::Slice(VX, torch::indexing::None)});
        auto vel_dim = vel.size(-1);
        auto center = anchor.index({"...", torch::indexing::Slice({X, Y, Z})});
        auto size = anchor.index({"...", torch::indexing::Slice({W, L, H})});
        auto yaw = anchor.index({"...", torch::indexing::Slice({COS_YAW, SIN_YAW})});

        torch::Tensor time_interval;
        if (!time_intervals.empty() && i < time_intervals.size()) {
            time_interval = time_intervals[i];
        } else if (src_timestamp.defined() && !dst_timestamps.empty() && i < dst_timestamps.size()) {
            time_interval = (src_timestamp - dst_timestamps[i]).to(vel.dtype());
        }

        if (time_interval.defined()) {
            // Object motion compensation
            auto translation = vel.transpose(0, -1) * time_interval;
            translation = translation.transpose(0, -1);

            // Apply translation to center
            if (center.dtype() != translation.dtype()) {
                translation = translation.to(center.dtype());
            }
            center = center - translation;
        }

        if (i < T_src2dst_list.size() && T_src2dst_list[i].defined()) {
            // Ego-motion compensation
            auto T_src2dst = T_src2dst_list[i].to(anchor.dtype()).unsqueeze(1);

            // Apply transformation to center
            center = torch::matmul(
                T_src2dst.index({"...", torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}),
                center.unsqueeze(-1)
            ).squeeze(-1) + T_src2dst.index({"...", torch::indexing::Slice(0, 3), 3});

            // Apply transformation to yaw
            yaw = torch::matmul(
                T_src2dst.index({"...", torch::indexing::Slice(0, 2), torch::indexing::Slice(0, 2)}),
                yaw.unsqueeze(-1)
            ).squeeze(-1);

            // Apply transformation to velocity
            vel = torch::matmul(
                T_src2dst.index({"...", torch::indexing::Slice(0, vel_dim), torch::indexing::Slice(0, vel_dim)}),
                vel.unsqueeze(-1)
            ).squeeze(-1);
        }

        // Reorder yaw components
        auto yaw_reordered = yaw.index({"...", torch::indexing::Slice({1, 0})});

        // Concatenate to create destination anchors
        auto dst_anchor = torch::cat({center, size, yaw_reordered, vel}, -1);
        dst_anchors.push_back(dst_anchor);
    }

    return dst_anchors;
}

/**
 * @brief Calculate Euclidean distance of anchors from origin in 2D
 *
 * @param anchor Input anchor tensor
 * @return Tensor of distances
 */
torch::Tensor SparseBox3DKeyPointsGenerator::distance(const torch::Tensor& anchor) {
    return torch::norm(anchor.index({"...", torch::indexing::Slice(0, 2)}), 2, -1);
}
