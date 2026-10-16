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

#ifndef INSTANCE_BANK_H
#define INSTANCE_BANK_H

#include <torch/torch.h>

#include "../utils/log_utils.h"
#include "../utils/tensor.h"

/**
 * @brief Interface class for anchor projection operations
 *
 * This abstract class defines the interface for anchor projection operations
 * that transform anchors based on time and coordinate transformations.
 */
class AnchorHandler {
public:
    virtual ~AnchorHandler() = default;

    /**
     * @brief Project anchors using transformation matrices and time intervals
     *
     * @param anchors Input anchors to project
     * @param transforms Transformation matrices for coordinate transforms
     * @param time_intervals Time intervals for temporal projection
     * @return Vector of projected anchor tensors
     */
    virtual std::vector<torch::Tensor> anchor_projection(
        const torch::Tensor& anchors,
        const std::vector<torch::Tensor>& transforms,
        const std::vector<torch::Tensor>& time_intervals) = 0;
};

// Type definitions for metadata structures
// Use std::map for now for simplicity since it works
typedef std::map<std::string, torch::jit::IValue> MetaData;
typedef std::map<std::string, torch::Tensor> DNMetaData;

/**
 * @brief InstanceBank class for managing and tracking object instances across frames
 *
 * This class implements functionality to maintain and update feature representations
 * of object instances across time, supporting object tracking and temporal association.
 */
class InstanceBank : public torch::nn::Module {
public:
    /**
     * @brief Constructor for InstanceBank
     *
     * @param num_anchor Maximum number of anchors to use
     * @param embed_dims Dimension of the embedding features
     * @param anchor Initial anchor tensor
     * @param anchor_handler Optional handler for anchor projection
     * @param num_temp_instances Number of temporary instances to track
     * @param default_time_interval Default time interval between frames
     * @param confidence_decay Decay factor for confidence scores
     * @param anchor_grad Whether to enable gradients for anchor parameters
     * @param feat_grad Whether to enable gradients for feature parameters
     * @param max_time_interval Maximum time interval to consider valid
     * @param reid_dims Optional re-identification feature dimension
     * @param gpu_id Device ID to use for computations
     */
    InstanceBank(
        int num_anchor,
        int embed_dims,
        const torch::Tensor& anchor,
        AnchorHandler* anchor_handler = nullptr,
        int num_temp_instances = 0,
        float default_time_interval = 0.5,
        float confidence_decay = 0.6,
        bool anchor_grad = true,
        bool feat_grad = true,
        float max_time_interval = 2.0,
        int reid_dims = -1,
        int gpu_id = -1);

    /**
     * @brief Initialize weights with proper initialization schemes
     */
    void init_weight();

    /**
     * @brief Reset internal state
     */
    void reset();

    /**
     * @brief Get features, anchors and time intervals for the current batch
     *
     * @param batch_size Size of the batch
     * @param metas Metadata for the current frames
     * @param dn_metas Optional denoising metadata
     * @return Tuple of tensors: instance_feature, anchor, cached_feature, cached_anchor, time_interval, reid_feature
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    get(int batch_size, const MetaData& metas, const DNMetaData* dn_metas = nullptr);

    /**
     * @brief Update instance features and anchors
     *
     * @param instance_feature Current instance features
     * @param anchor Current anchors
     * @param confidence Confidence scores
     * @return Updated instance features and anchors
     */
    std::tuple<torch::Tensor, torch::Tensor>
    update(torch::Tensor instance_feature, torch::Tensor anchor, torch::Tensor confidence);

    /**
     * @brief Cache features, anchors and metadata for temporal association
     *
     * @param instance_feature Instance features to cache
     * @param anchor Anchors to cache
     * @param confidence Confidence scores
     * @param metas Optional metadata
     * @param feature_maps Optional feature maps
     */
    void cache(
        torch::Tensor instance_feature,
        torch::Tensor anchor,
        torch::Tensor confidence,
        const MetaData* metas = nullptr,
        const std::vector<torch::Tensor>* feature_maps = nullptr);

    /**
     * @brief Get instance IDs based on confidence scores
     *
     * @param confidence Confidence scores for instances
     * @param anchor Optional anchor tensor
     * @param threshold Optional confidence threshold (negative value means no threshold)
     * @return Tensor of instance IDs
     */
    torch::Tensor get_instance_id(
        torch::Tensor confidence,
        torch::Tensor anchor = torch::Tensor(),
        float threshold = -1.0);

    /**
     * @brief Update instance IDs based on confidence scores
     *
     * @param instance_id_tensor Current instance IDs
     * @param confidence Optional confidence scores
     */
    void update_instance_id(
        torch::Tensor instance_id_tensor,
        torch::Tensor confidence = torch::Tensor());

    /**
     * @brief Get the mask tensor
     *
     * @return The internal mask tensor
     */
    torch::Tensor getMask() const {
        return m_mask;
    }

private:
    // Parameters
    int m_embed_dims;
    int m_reid_dims;
    int m_num_anchor;
    int m_num_temp_instances;
    float m_default_time_interval;
    float m_confidence_decay;
    float m_max_time_interval;
    AnchorHandler* m_anchor_handler;

    // Registered parameters
    torch::Tensor m_anchor;
    torch::Tensor m_anchor_init;
    torch::Tensor m_instance_feature;
    torch::Tensor m_reid_feature;

    // Internal state
    torch::Tensor m_cached_feature;
    torch::Tensor m_cached_anchor;
    MetaData m_metas;
    torch::Tensor m_mask;
    torch::Tensor m_confidence;
    torch::Tensor m_temp_confidence;
    torch::Tensor m_instance_id;
    int64_t m_prev_id;

    // Device to use for computations
    torch::Device m_device;
};

/**
 * @brief Load anchor data from a NumPy .npy file
 *
 * @param filename Path to the .npy file containing anchor data
 * @return torch::Tensor Loaded anchor tensor
 */
torch::Tensor load_anchor(const std::string& filename);

#endif // INSTANCE_BANK_H
