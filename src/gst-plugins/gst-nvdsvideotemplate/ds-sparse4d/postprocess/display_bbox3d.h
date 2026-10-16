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

#ifndef DISPLAY_BBOX3D_H
#define DISPLAY_BBOX3D_H

#include <string>
#include <vector>
#include <torch/torch.h>
#include "gstnvdsmeta.h"
// Forward declaration to avoid circular dependency
struct DecodedResult;

#include "../utils/log_utils.h"

/**
 * @brief Structure to hold class label and its confidence threshold
 */
struct ClassThreshold {
    std::string label;        ///< Class label name
    float threshold;          ///< Confidence threshold for this class
};

/**
 * @brief Fetch class labels and thresholds from a labels file
 *
 * This function reads class labels and their confidence thresholds from a file 
 * where each entry is in format: class_name:threshold;
 * Example: Person:0.85;Vehicle:0.90;
 *
 * @param labels_file Path to the file containing class labels and thresholds
 * @param logger Logger for debug and warning messages
 * @return Vector of ClassThreshold structures containing labels and thresholds
 */
std::vector<ClassThreshold> fetch_class_labels_with_thresholds(
    const std::string& labels_file,
    PrettyLogger* logger
);

/**
 * @brief Extract class label strings from ClassThreshold vector
 *
 * @param class_thresholds Vector of ClassThreshold structures
 * @return Vector of class label strings only
 */
std::vector<std::string> extract_class_labels(
    const std::vector<ClassThreshold>& class_thresholds
);

/**
 * @brief Attach 3D bounding boxes to display metadata for visualization
 *
 * This function processes 3D bounding box projections and attaches them to
 * the DeepStream display metadata for visualization on the frame.
 *
 * @param batch_meta DeepStream batch metadata
 * @param frame_meta DeepStream frame metadata
 * @param camera_frames Tensor containing projected 3D bbox coordinates
 * @param boxes_3d Tensor containing 3D bounding box coordinates
 * @param labels_3d Tensor containing label indices
 * @param instance_ids Tensor containing instance IDs
 * @param scores_3d Tensor containing confidence scores
 * @param img_width Width of the image
 * @param img_height Height of the image
 * @param class_labels Vector of class label strings
 * @param bbox_color_mode Color mode for bounding boxes (class_id or instance_id)
 * @param display_class_labels Whether to display class labels
 * @param display_instance_ids Whether to display instance IDs
 * @param display_class_score Whether to display class confidence score
 * @param logger Logger for debug and warning messages
 */
void attach_bbox3d_to_display(
    NvDsBatchMeta* batch_meta,
    NvDsFrameMeta* frame_meta,
    const torch::Tensor& camera_frames,
    const torch::Tensor& boxes_3d,
    const torch::Tensor& labels_3d,
    const torch::Tensor& instance_ids,
    const torch::Tensor& scores_3d,
    int img_width,
    int img_height,
    const std::vector<std::string>& class_labels,
    const std::string& bbox_color_mode,
    const bool& display_class_labels,
    const bool& display_instance_ids,
    const bool& display_class_score,
    PrettyLogger* logger
);

/**
 * @brief Clamps a point to stay within image boundaries
 *
 * @param x X-coordinate to clamp
 * @param y Y-coordinate to clamp
 * @param img_width Image width
 * @param img_height Image height
 */
void clamp_point(float* x, float* y, int img_width, int img_height);

/**
 * @brief Calculates the centroid of a 3D bounding box from its 2D projection
 *
 * @param points Array of 2D points
 * @param centroid Array to store centroid coordinates
 */
void calculate_centroid(const float* points, float* centroid);

/**
 * @brief Generates a consistent color based on a class label
 *
 * @param label Class label string
 * @param red Output red component [0-1]
 * @param green Output green component [0-1]
 * @param blue Output blue component [0-1]
 * @param num_classes Number of unique classes
 */
void get_color_from_label(const std::string& label, float* red, float* green, float* blue, int num_classes);

/**
 * @brief Generates a color based on instance ID for tracking visualization
 *
 * @param instance_id Instance ID
 * @param red Output red component [0-1]
 * @param green Output green component [0-1]
 * @param blue Output blue component [0-1]
 */
void get_color_from_instance_id(int64_t instance_id, float* red, float* green, float* blue);

/**
 * @brief Counts visible corners of a 3D bounding box
 *
 * @param points Array of 2D points
 * @param num_corners Number of corners
 * @param img_width Image width
 * @param img_height Image height
 * @return true if half or more corners are visible
 */
bool count_visible_corners(const float* points, int num_corners, int img_width, int img_height);

/**
 * @brief Attach 2D bounding boxes as NvDsObjectMeta to frame metadata
 *
 * @param batch_meta DeepStream batch metadata
 * @param frame_meta DeepStream frame metadata
 * @param bbox2d Tensor containing 2D bounding boxes [N, 4] where 4 = [x, y, w, h]
 * @param boxes_3d Tensor containing 3D bounding box information
 * @param labels_3d Tensor containing class labels
 * @param instance_ids Tensor containing instance IDs for tracking
 * @param scores_3d Tensor containing confidence scores
 * @param class_labels Vector of class label strings
 * @param bbox_color_mode Color mode: "class_id" for class-based or "instance_id" for instance-based colors
 * @param display_2d_bbox Whether to display 2D bounding boxes on frames
 * @param logger Logger for debug and warning messages
 */
 void attach_bbox2d_to_objectmeta(
    NvDsBatchMeta* batch_meta,
    NvDsFrameMeta* frame_meta,
    const torch::Tensor& bbox2d,
    const torch::Tensor& boxes_3d,
    const torch::Tensor& labels_3d,
    const torch::Tensor& instance_ids,
    const torch::Tensor& scores_3d,
    const std::vector<std::string>& class_labels,
    const std::string& bbox_color_mode,
    const bool& display_2d_bbox,
    PrettyLogger* logger
);

#endif // DISPLAY_BBOX3D_H