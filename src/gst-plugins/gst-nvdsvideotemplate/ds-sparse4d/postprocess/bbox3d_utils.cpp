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
#include <cmath>
#include <algorithm>
#include "bbox3d_utils.h"

// Constants for box3d tensor indices
constexpr int X = 0;
constexpr int Y = 1;
constexpr int Z = 2;
constexpr int W = 3;  // width
constexpr int L = 4;  // length
constexpr int H = 5;  // height
constexpr int YAW = 6;

torch::Tensor box3d_to_corners(const torch::Tensor& box3d) {
    // Get batch size (number of boxes)
    int64_t num_boxes = box3d.size(0);

    // Create normalized corner coordinates (8 corners per box)
    // This creates a pattern of corners at (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), etc.
    std::vector<std::vector<float>> corners_norm_data = {
        {-0.5, -0.5, -0.5}, {-0.5, -0.5, 0.5}, {-0.5, 0.5, 0.5}, {-0.5, 0.5, -0.5},
        {0.5, -0.5, -0.5}, {0.5, -0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, -0.5}
    };

    // Create tensor on CPU
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor corners_norm = torch::zeros({8, 3}, options);

    // Direct access for CPU tensors
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 3; j++) {
            corners_norm[i][j] = corners_norm_data[i][j];
        }
    }

    // Reshape corners_norm to [1, 8, 3] for broadcasting
    corners_norm = corners_norm.unsqueeze(0);

    // Scale corners by box dimensions (W, L, H)
    auto dimensions = box3d.index({torch::indexing::Slice(), torch::indexing::Slice(W, H+1)});
    dimensions = dimensions.unsqueeze(1);  // [N, 1, 3]

    auto corners = dimensions * corners_norm;  // [N, 8, 3]

    // Create rotation matrices for each box based on yaw angle
    auto rot_sin = torch::sin(box3d.index({torch::indexing::Slice(), YAW}));
    auto rot_cos = torch::cos(box3d.index({torch::indexing::Slice(), YAW}));

    // Initialize rotation matrices (identity matrix for each box)
    auto rot_mat = torch::eye(3, options).unsqueeze(0).repeat({num_boxes, 1, 1});

    // Fill in rotation matrix values for rotating around Z-axis
    rot_mat.index_put_({torch::indexing::Slice(), 0, 0}, rot_cos);
    rot_mat.index_put_({torch::indexing::Slice(), 0, 1}, -rot_sin);
    rot_mat.index_put_({torch::indexing::Slice(), 1, 0}, rot_sin);
    rot_mat.index_put_({torch::indexing::Slice(), 1, 1}, rot_cos);

    // Apply rotation to corners: [N, 3, 3] @ [N, 8, 3, 1] -> [N, 8, 3]
    corners = corners.unsqueeze(-1);  // [N, 8, 3, 1]
    corners = torch::matmul(rot_mat.unsqueeze(1), corners).squeeze(-1);  // [N, 8, 3]

    // Translate corners by adding box center coordinates (X, Y, Z)
    auto centers = box3d.index({torch::indexing::Slice(), torch::indexing::Slice(X, Z+1)});
    centers = centers.unsqueeze(1);  // [N, 1, 3]

    corners = corners + centers;  // [N, 8, 3]

    return corners;
}

torch::Tensor generate_bbox3d_on_camera_frames(
    const torch::Tensor& bboxes3d,
    const torch::Tensor& projection_matrices) {

    // Convert 3D bounding boxes to corner points
    torch::Tensor corners_3d = box3d_to_corners(bboxes3d);
    int64_t num_bboxes = corners_3d.size(0);

    // Create homogeneous coordinates by appending ones
    // Reshape to [N*8, 3]
    auto corners_flat = corners_3d.reshape({-1, 3});
    auto ones = torch::ones({corners_flat.size(0), 1}, corners_flat.options());
    auto pts_4d = torch::cat({corners_flat, ones}, 1);  // [N*8, 4]

    // Number of cameras
    int64_t num_cams = projection_matrices.size(0);

    // List to store projected 2D corners for each camera
    std::vector<torch::Tensor> projected_2d_corners;

    // Process each camera's projection matrix
    for (int64_t i = 0; i < num_cams; i++) {
        // Get current projection matrix
        auto proj_mat = projection_matrices[i].reshape({4, 4});

        // Project 3D points to the image plane: [N*8, 4] @ [4, 4]T -> [N*8, 4]
        auto pts_2d = torch::matmul(pts_4d, proj_mat.transpose(0, 1));

        // Normalize the projected points
        // Clip depth values to avoid division by zero
        auto depth = pts_2d.index({torch::indexing::Slice(), 2});
        depth = torch::clamp(depth, 1e-5, 1e5);

        // Normalize x, y by depth
        pts_2d.index({torch::indexing::Slice(), 0}) =
            pts_2d.index({torch::indexing::Slice(), 0}) / depth;
        pts_2d.index({torch::indexing::Slice(), 1}) =
            pts_2d.index({torch::indexing::Slice(), 1}) / depth;

        // Extract x, y coordinates and reshape to [N, 8, 2]
        auto img_corners = pts_2d.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)})
                              .reshape({num_bboxes, 8, 2});

        // Add to result list
        projected_2d_corners.push_back(img_corners);
    }

    // Stack all camera projections into a single tensor [num_cams, N, 8, 2]
    return torch::stack(projected_2d_corners);
}

/**
 * @brief Generate 2D bounding boxes from projected 3D rectangle corners
 *
 * This function creates 2D bounding boxes from projected 3D corners using
 * a simple min/max approach. It includes visibility checking to filter out
 * objects that are not sufficiently visible in the frame.
 * @param[in] rect3d Torch tensor containing projected 3D rectangle corners
 *                   Shape: [num_cams, N, 8, 2] where:
 *                   - num_cams: number of camera views
 *                   - N: number of objects
 *                   - 8: eight corners of the 3D bounding box
 *                   - 2: x, y coordinates in image space
 * @param[in] image_width Width of the image frame (default: 1920)
 * @param[in] image_height Height of the image frame (default: 1080)
 *
 * @return Torch tensor of 2D bounding boxes
 *         Shape: [num_cams, N, 4] where 4 represents [x, y, w, h]
 *         - x, y: top-left corner coordinates
 *         - w, h: width and height of the bounding box
 *         - Invalid boxes have negative width/height for filtering
 */
torch::Tensor generate_bbox2d_from_rect3d(const torch::Tensor& rect3d,
                                          int image_width,
                                          int image_height) {
    // Validate input shape
    if (rect3d.dim() != 4) {
        throw std::runtime_error("rect3d must be a 4D tensor with shape [num_cams, N, 8, 2]");
    }

    int64_t num_cams = rect3d.size(0);

    // Configuration parameters
    const int MIN_VISIBLE_CORNERS = 4;  // Minimum corners that must be visible
    const float MIN_VISIBLE_RATIO = 0.5f;  // Minimum ratio of visible corners (4/8 = 0.5)

    // List to store 2D bounding boxes for each camera
    std::vector<torch::Tensor> bbox2d_list;

    // Process each camera view
    for (int64_t cam_idx = 0; cam_idx < num_cams; cam_idx++) {
        // Get corners for current camera: [N, 8, 2]
        auto cam_corners = rect3d[cam_idx];
        int64_t num_objects = cam_corners.size(0);

        // Prepare output tensors
        auto options = cam_corners.options();
        auto bbox_x = torch::zeros({num_objects}, options);
        auto bbox_y = torch::zeros({num_objects}, options);
        auto bbox_w = torch::zeros({num_objects}, options);
        auto bbox_h = torch::zeros({num_objects}, options);

        // Process each object
        for (int64_t obj_idx = 0; obj_idx < num_objects; obj_idx++) {
            // Get corners for current object: [8, 2]
            auto obj_corners = cam_corners[obj_idx];

            // First, check visibility of the 3D object
            // Count how many corners are within the image boundaries
            int visible_corner_count = 0;
            bool has_extreme_values = false;

            for (int i = 0; i < 8; i++) {
                float x = obj_corners[i][0].item<float>();
                float y = obj_corners[i][1].item<float>();

                // Check for extreme/invalid values that indicate poor projection
                if (std::abs(x) > image_width * 10 || std::abs(y) > image_height * 10 ||
                    std::isnan(x) || std::isnan(y) || std::isinf(x) || std::isinf(y)) {
                    has_extreme_values = true;
                    break;
                }

                // Count corners within image bounds (with small margin)
                if (x >= -image_width * 0.1f && x <= image_width * 1.1f &&
                    y >= -image_height * 0.1f && y <= image_height * 1.1f) {
                    visible_corner_count++;
                }
            }

            // Check if object is sufficiently visible
            bool is_visible = !has_extreme_values &&
                            (visible_corner_count >= MIN_VISIBLE_CORNERS) &&
                            (visible_corner_count >= 8 * MIN_VISIBLE_RATIO);

            if (!is_visible) {
                // Mark this box as invalid by setting negative dimensions
                // This allows downstream processing to filter it out
                bbox_x[obj_idx] = 0;
                bbox_y[obj_idx] = 0;
                bbox_w[obj_idx] = -1;  // Negative width indicates invalid box
                bbox_h[obj_idx] = -1;  // Negative height indicates invalid box
                continue;
            }

            // Simple approach: Find min/max from all 8 corners
            // This gives us the complete 2D extent of the projected 3D box
            float x_min = 1e9f, y_min = 1e9f;
            float x_max = -1e9f, y_max = -1e9f;

            // Iterate through all 8 corners to find min/max
            for (int i = 0; i < 8; i++) {
                float x = obj_corners[i][0].item<float>();
                float y = obj_corners[i][1].item<float>();

                x_min = std::min(x_min, x);
                y_min = std::min(y_min, y);
                x_max = std::max(x_max, x);
                y_max = std::max(y_max, y);
            }

            // Calculate width and height
            float width = x_max - x_min;
            float height = y_max - y_min;

            // Store the bounding box
            bbox_x[obj_idx] = x_min;
            bbox_y[obj_idx] = y_min;
            bbox_w[obj_idx] = width;
            bbox_h[obj_idx] = height;
        }

        // Stack to create [N, 4] tensor with format [x, y, w, h]
        auto bbox2d = torch::stack({bbox_x, bbox_y, bbox_w, bbox_h}, /*dim=*/1);  // [N, 4]

        // Add to result list
        bbox2d_list.push_back(bbox2d);
    }
    // Stack all camera bounding boxes into a single tensor [num_cams, N, 4]
    return torch::stack(bbox2d_list);
}

void test_generate_bbox3d_on_camera_frames(PrettyLogger* logger) {
    logger->info("Test: Generate BBox3d on Camera Frames with Sequential Data");

    int64_t num_objects = 2;
    int64_t num_cameras = 2;

    // For bboxes3d: num_objects X 12 with sequential data
    torch::Tensor bboxes3d = torch::zeros({num_objects, 12});
    for (int64_t i = 0; i < num_objects; i++) {
        for (int64_t j = 0; j < 12; j++) {
            bboxes3d[i][j] = i * 12 + j + 1;
        }
    }

    // For projection_matrices num_cameras X 4 X 4 with sequential data
    torch::Tensor projection_matrices = torch::zeros({num_cameras, 4, 4});
    for (int64_t i = 0; i < num_cameras; i++) {
        for (int64_t j = 0; j < 4; j++) {
            for (int64_t k = 0; k < 4; k++) {
                projection_matrices[i][j][k] = i * 16 + j * 4 + k + 1;
            }
        }
    }

    // logger->info("bboxes3d: " + torch_tensor_sizes_to_string(bboxes3d) + "  " + torch_tensor_to_string(bboxes3d, -1));
    // logger->info("projection_matrices: " +torch_tensor_sizes_to_string(projection_matrices) + "  " + torch_tensor_to_string(projection_matrices, -1));

    // Generate bbox3d on camera frames
    torch::Tensor rect_3d = generate_bbox3d_on_camera_frames(bboxes3d, projection_matrices);
    // logger->info("rect_3d: " + torch_tensor_sizes_to_string(rect_3d) + "  " + torch_tensor_to_string(rect_3d, -1));

    // The expected output is
    torch::Tensor expected_output = torch::tensor({
        0.267706, 0.633853, 0.272054, 0.636027, 0.296262, 0.648131, 0.380885, 0.690443,
        0.172972, 0.586486, 0.223756, 0.611878, 0.241428, 0.620714, 0.212647, 0.606323,
        0.2073, 0.60365, 0.24164, 0.62082, 0.233972, 0.616986, 0.210317, 0.605159,
        0.167168, 0.583584, 0.205616, 0.602808, 0.207392, 0.603696, 0.181032, 0.590516,
        0.702874, 0.851437, 0.703592, 0.851796, 0.707686, 0.853843, 0.723391, 0.861695,
        0.688391, 0.844195, 0.695887, 0.847944, 0.698638, 0.849319, 0.694197, 0.847098,
        0.693394, 0.846697, 0.698671, 0.849336, 0.697468, 0.848734, 0.693846, 0.846923,
        0.687571, 0.843785, 0.693142, 0.846571, 0.693407, 0.846704, 0.689542, 0.844771
    }, torch::kFloat32).reshape({2, 2, 8, 2});

    // Check if the output matches the expected output with a small tolerance for floating-point differences
    // Use allclose instead of equal to handle floating-point precision issues
    if (torch::allclose(rect_3d, expected_output, 1e-5, 1e-5)) {
        logger->info("Test: Generate BBox3d on Camera Frames with Sequential Data Passed");
    } else {
        logger->info("Test: Generate BBox3d on Camera Frames with Sequential Data Failed");
        // Print the absolute difference to help debug
        logger->info("Max absolute difference: " +
                     std::to_string(torch::max(torch::abs(rect_3d - expected_output)).item<float>()));
    }
}
