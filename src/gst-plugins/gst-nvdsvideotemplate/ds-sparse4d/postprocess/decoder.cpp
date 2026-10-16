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

#include "decoder.h"
#include "display_bbox3d.h"
#include <unordered_map>

// Constants for box indices (matching Python implementation)
constexpr int X = 0;
constexpr int Y = 1;
constexpr int Z = 2;
constexpr int W = 3;
constexpr int L = 4;
constexpr int H = 5;
constexpr int SIN_YAW = 6;
constexpr int COS_YAW = 7;
constexpr int VX = 8;
constexpr int VY = 9;
constexpr int VZ = 10;
constexpr int CNS = 0; // Centerness index in quality tensor

SparseBox3DDecoder::SparseBox3DDecoder(
    int num_output,
    float score_threshold,
    bool sorted,
    torch::Tensor group_origin,
    int gpu_id,
    PrettyLogger* logger)
    : num_output(num_output),
      score_threshold(score_threshold),
      sorted(sorted),
      group_origin(group_origin),
      gpu_id(gpu_id),
      logger(logger) {
        logger->info("SparseBox3DDecoder initialized with parameters:");
        logger->info("  num_output: " + std::to_string(num_output));
        logger->info("  score_threshold: " + std::to_string(score_threshold));
        logger->info("  sorted: " + std::to_string(sorted));
        logger->info("  group_origin: " + std::to_string(group_origin.index({0}).item<float>()) + ", " + std::to_string(group_origin.index({1}).item<float>()));
        logger->info("  gpu_id: " + std::to_string(gpu_id));
}

torch::Tensor SparseBox3DDecoder::decode_box(torch::Tensor& box, bool dummy_rotation) {
    // Add group origin to box to account for recentering
    box.index({torch::indexing::Slice(), X}) += group_origin.index({0});
    box.index({torch::indexing::Slice(), Y}) += group_origin.index({1});

    // Calculate z_rotation from sin and cos components
    torch::Tensor z_rotation = torch::atan2(
        box.index({torch::indexing::Slice(), SIN_YAW}), 
        box.index({torch::indexing::Slice(), COS_YAW}));

    // Create zero tensors for x and y rotation
    torch::Tensor x_rotation = torch::zeros_like(z_rotation);
    torch::Tensor y_rotation = torch::zeros_like(z_rotation);

    // Get components with select or narrow instead of index with list
    torch::Tensor pos = box.narrow(1, X, 3);       // X,Y,Z
    torch::Tensor dim = box.narrow(1, W, 3).exp(); // W,L,H
    torch::Tensor vel = box.narrow(1, VX, box.size(1) - VX); // Velocity components

    if (dummy_rotation) {
        // Include dummy x and y rotation values
        return torch::cat({
            pos, dim,
            x_rotation.unsqueeze(1),
            y_rotation.unsqueeze(1),
            z_rotation.unsqueeze(1),
            vel
        }, 1);
    } else {
        // Without dummy rotation values
        return torch::cat({
            pos, dim,
            z_rotation.unsqueeze(1),
            vel
        }, 1);
    }
}

std::vector<DecodedResult> SparseBox3DDecoder::decode(
    const std::vector<torch::Tensor>& cls_scores,
    const std::vector<torch::Tensor>& box_preds,
    const torch::Tensor& instance_id,
    const torch::Tensor& instance_feature,
    const std::vector<torch::Tensor>& quality,
    int output_idx) {

    // Find the last valid output_idx
    if (output_idx == -1) {
        for (int i = cls_scores.size() - 1; i >= 0; i--) {
            if (cls_scores[i].defined() && box_preds[i].defined()) {
                output_idx = i;
                break;
            }
        }
    }

    if (output_idx < 0 ||
        static_cast<size_t>(output_idx) >= cls_scores.size() ||
        static_cast<size_t>(output_idx) >= box_preds.size()) {
        logger->error("Invalid output_idx: " + std::to_string(output_idx));
        return {};
    }

    // Check if we need to squeeze class dimension
    bool squeeze_cls = instance_id.defined();

    // Apply sigmoid to cls_scores scores
    torch::Tensor cls_score = cls_scores[output_idx];
    cls_score.sigmoid_();
    torch::Tensor box_pred = box_preds[output_idx];

    // Handle squeezing of class dimension if instance_id is provided
    torch::Tensor cls_ids;
    if (squeeze_cls) {
        // Get max score and corresponding class ID
        auto max_vals = cls_score.max(2);
        cls_score = std::get<0>(max_vals);
        cls_ids = std::get<1>(max_vals);
        cls_score = cls_score.unsqueeze(-1);
    }

    // Get batch size and other dimensions
    auto bs = cls_score.size(0);
    // auto num_pred = cls_score.size(1);
    auto num_cls = cls_score.size(2);

    // Flatten and get top-k scores
    auto flattened = cls_score.flatten(1);
    auto topk_result = flattened.topk(num_output, 1, sorted);
    cls_score = std::get<0>(topk_result);
    auto indices = std::get<1>(topk_result);

    // If not squeezed, get class IDs from indices
    if (!squeeze_cls) {
        cls_ids = indices % num_cls;
    }

    torch::Tensor centerness;
    torch::Tensor cls_scores_origin;
    // Handle quality/centerness if provided
    if (!quality.empty() && static_cast<size_t>(output_idx) < quality.size()) {
        centerness = quality[output_idx].index({torch::indexing::Slice(), torch::indexing::Slice(), CNS});

        // Gather centerness values using indices
        auto gather_indices = (indices / num_cls).to(torch::kInt64);
        centerness = torch::gather(centerness, 1, gather_indices);

        // Store original scores before applying centerness
        cls_scores_origin = cls_score.clone();

        // Apply centerness sigmoid to scores
        cls_score *= centerness.sigmoid();

        // Sort by updated scores
        auto sort_result = torch::sort(cls_score, 1, true);
        cls_score = std::get<0>(sort_result);
        auto idx = std::get<1>(sort_result);

        // Update class IDs and indices after sorting
        if (!squeeze_cls) {
            cls_ids = torch::gather(cls_ids, 1, idx);
        }

        if (score_threshold > 0.0f) {
            cls_scores_origin = torch::gather(cls_scores_origin, 1, idx);
        }

        indices = torch::gather(indices, 1, idx);
    }

    // Apply score threshold if specified
    torch::Tensor mask;
    if (score_threshold > 0.0f) {
        mask = cls_score >= score_threshold;
    }

    // Process each sample in the batch
    std::vector<DecodedResult> output;

    for (int i = 0; i < bs; i++) {
        DecodedResult result;

        // Get category IDs for this sample
        auto category_ids = cls_ids[i];

        // If squeezing, need to gather from original indices
        if (squeeze_cls) {
            category_ids = category_ids.index({indices[i]});
        }

        // Get scores and boxes
        auto scores = cls_score[i];
        auto box = box_pred.index({i, (indices[i] / num_cls).to(torch::kInt64)});

        // Apply threshold mask if needed
        if (score_threshold > 0.0f && mask.defined()) {
            // If no boxes above threshold, skip this sample
            if (mask[i].sum().item<int>() == 0) {
                continue;
            }

            // Filter by mask directly, matching Python implementation
            category_ids = category_ids.index({mask[i]});
            scores = scores.index({mask[i]});
            box = box.index({mask[i]});
        }

        if (!quality.empty()) {
            auto scores_origin = cls_scores_origin[i];
            if (score_threshold > 0.0f && mask.defined()) {
                scores_origin = scores_origin.index({mask[i]});
            }
            result.cls_scores = scores_origin;
        }

        // Decode box parameters
        auto decoded_box = decode_box(box);

        // Set result fields
        result.boxes_3d = decoded_box;
        result.scores_3d = scores;
        result.labels_3d = category_ids;

        // Add instance information if provided
        if (instance_id.defined()) {
            auto ids = instance_id.index({i, (indices[i] / num_cls).to(torch::kInt64)});
            if (score_threshold > 0.0f && mask.defined()) {
                ids = ids.index({mask[i]});
            }
            result.instance_ids = ids;
        }

        if (instance_feature.defined()) {
            auto feats = instance_feature.index({i, (indices[i] / num_cls).to(torch::kInt64)});
            if (score_threshold > 0.0f && mask.defined()) {
                feats = feats.index({mask[i]});
            }
            result.instance_feats = feats;
        }

        output.push_back(result);
    }

    if (!output.empty() && output[0].boxes_3d.defined()) {
        logger->info("Found "+ std::to_string(output.size()) + " decoded batch with " + std::to_string(output[0].boxes_3d.size(0)) + " detections above threshold " + std::to_string(score_threshold));
    } else {
        logger->warning("No BBoxes found above threshold " + std::to_string(score_threshold) + " after Decoding!");
    }

    return output;
}

void filter_results(const std::vector<DecodedResult>& results, 
                   const std::vector<ClassThreshold>& class_thresholds, 
                   PrettyLogger* logger) {
    // Build a threshold lookup map for efficient access
    std::unordered_map<int64_t, float> threshold_map;
    for (size_t i = 0; i < class_thresholds.size(); ++i) {
        threshold_map[static_cast<int64_t>(i)] = class_thresholds[i].threshold;
    }

    for (auto& result : const_cast<std::vector<DecodedResult>&>(results)) {
        // Check if required tensors are properly defined and have valid shapes
        if (!result.scores_3d.defined() || !result.labels_3d.defined() || 
            result.scores_3d.numel() == 0 || result.labels_3d.numel() == 0) {
            continue;
        }

        // Ensure tensors are 1D and have the same size
        if (result.scores_3d.dim() != 1 || result.labels_3d.dim() != 1 ||
            result.scores_3d.size(0) != result.labels_3d.size(0)) {
            logger->warning("Invalid tensor dimensions in filter_results, skipping");
            continue;
        }

        int64_t num_detections = result.scores_3d.size(0);
        if (num_detections == 0) {
            continue;
        }

        // Create mask using safer tensor operations
        torch::Tensor mask = torch::zeros({num_detections}, torch::kBool);
        if (result.scores_3d.device() != torch::kCPU) {
            mask = mask.to(result.scores_3d.device());
        }

        // Convert tensors to CPU for safe access
        torch::Tensor scores_cpu = result.scores_3d.to(torch::kCPU);
        torch::Tensor labels_cpu = result.labels_3d.to(torch::kCPU);
        torch::Tensor mask_cpu = mask.to(torch::kCPU);

        float* scores_data = scores_cpu.data_ptr<float>();
        int64_t* labels_data = labels_cpu.data_ptr<int64_t>();
        bool* mask_data = mask_cpu.data_ptr<bool>();

        int64_t valid_count = 0;

        // Check each detection against its class-specific threshold
        for (int64_t i = 0; i < num_detections; ++i) {
            int64_t class_id = labels_data[i];
            float score = scores_data[i];
            
            // Get threshold for this class, use default 0.5 if not found
            float threshold = 0.5f;
            std::string class_name = "UNKNOWN";
            auto it = threshold_map.find(class_id);
            if (it != threshold_map.end()) {
                threshold = it->second;
                // Find class name by searching through class_thresholds
                for (size_t j = 0; j < class_thresholds.size(); ++j) {
                    if (static_cast<int64_t>(j) == class_id) {
                        class_name = class_thresholds[j].label;
                        break;
                    }
                }
            } else {
                logger->warning("No threshold found for class ID " + std::to_string(class_id) + 
                              ", using default threshold 0.5");
            }

            // Apply class-specific threshold
            if (score >= threshold) {
                mask_data[i] = true;
                valid_count++;
            }
        }

        // Move mask back to original device if needed
        if (result.scores_3d.device() != torch::kCPU) {
            mask = mask_cpu.to(result.scores_3d.device());
        } else {
            mask = mask_cpu;
        }

        // Skip if no valid elements
        if (valid_count == 0) {
            // Clear all tensors when no valid detections remain
            result.boxes_3d = torch::Tensor();
            result.scores_3d = torch::Tensor();
            result.cls_scores = torch::Tensor();
            result.labels_3d = torch::Tensor();
            result.instance_ids = torch::Tensor();
            result.instance_feats = torch::Tensor();
            continue;
        }

        // Apply mask to all tensors safely
        try {
            result.boxes_3d = result.boxes_3d.index({mask});
            result.scores_3d = result.scores_3d.index({mask});
            result.labels_3d = result.labels_3d.index({mask});

            if (result.cls_scores.defined()) {
                result.cls_scores = result.cls_scores.index({mask});
            }
            
            // Apply mask to instance_ids if defined
            if (result.instance_ids.defined()) {
                result.instance_ids = result.instance_ids.index({mask});
            }

            // Apply mask to instance_feats if defined
            if (result.instance_feats.defined()) {
                result.instance_feats = result.instance_feats.index({mask});
            }
        } catch (const std::exception& e) {
            logger->error("Error applying mask in filter_results: " + std::string(e.what()));
            continue;
        }
    }
    
    if (!results.empty() && results[0].boxes_3d.defined()) {
        logger->info("Filtered " + std::to_string(results.size()) + " decoded batch with " + 
                    std::to_string(results[0].boxes_3d.size(0)) + 
                    " detections above class-specific thresholds");
    } else {
        logger->warning("No BBoxes found above class-specific thresholds after filtering!");
    }
}