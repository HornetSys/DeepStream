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

#ifndef DECODER_H
#define DECODER_H

#include <torch/torch.h>

#include "../utils/log_utils.h"

// Forward declaration to avoid circular dependency
struct ClassThreshold;
#include "../utils/tensor.h"

// Structure to hold decoded result data
struct DecodedResult {
    torch::Tensor boxes_3d;
    torch::Tensor scores_3d;
    torch::Tensor labels_3d;
    torch::Tensor cls_scores;
    torch::Tensor instance_ids;
    torch::Tensor instance_feats;
};

class SparseBox3DDecoder {
public:
    /**
     * Constructor for SparseBox3DDecoder.
     *
     * @param num_output Maximum number of detections to output
     * @param score_threshold Confidence threshold for filtering detections
     * @param sorted Whether to sort the results by score
     * @param group_origin Group origin values
     * @param gpu_id GPU device ID to use
     * @param logger Logger instance for debug messages
     */
    SparseBox3DDecoder(
        int num_output = 300,
        float score_threshold = 0.0f,
        bool sorted = true,
        torch::Tensor group_origin = torch::Tensor(),
        int gpu_id = 0,
        PrettyLogger* logger = nullptr);

    /**
     * Decodes box parameters from network output format to actual box dimensions.
     *
     * @param box Input tensor containing encoded box parameters
     * @param dummy_rotation If true, adds dummy x and y rotation values
     * @return Decoded box parameters
     */
    torch::Tensor decode_box(torch::Tensor& box, bool dummy_rotation = true);

    /**
     * Decodes model predictions into final detection outputs.
     *
     * @param cls_scores Classification scores from the model
     * @param box_preds Box predictions from the model
     * @param instance_id Instance IDs (optional)
     * @param instance_feature Instance features (optional)
     * @param quality Quality scores including centerness (optional)
     * @param output_idx Output index to process
     * @return Vector of DecodedResult structures with detection results
     */
    std::vector<DecodedResult> decode(
        const std::vector<torch::Tensor>& cls_scores,
        const std::vector<torch::Tensor>& box_preds,
        const torch::Tensor& instance_id = torch::Tensor(),
        const torch::Tensor& instance_feature = torch::Tensor(),
        const std::vector<torch::Tensor>& quality = std::vector<torch::Tensor>(),
        int output_idx = -1);

private:
    int num_output;
    float score_threshold;
    bool sorted;
    torch::Tensor group_origin;
    int gpu_id;
    PrettyLogger* logger;
};

/**
 * Filters results based on per-class confidence thresholds.
 *
 * @param results Vector of DecodedResult structures containing detection results
 * @param class_thresholds Vector of ClassThreshold containing per-class confidence thresholds
 * @param logger Logger instance for debug messages
 */
void filter_results(const std::vector<DecodedResult>& results, 
                   const std::vector<ClassThreshold>& class_thresholds, 
                   PrettyLogger* logger);

#endif // DECODER_H