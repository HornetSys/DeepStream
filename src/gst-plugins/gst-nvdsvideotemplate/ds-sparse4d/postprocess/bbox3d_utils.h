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

#ifndef BBOX3D_UTILS_H
#define BBOX3D_UTILS_H

#include <vector>
#include <cmath>
#include <iostream>
#include <torch/torch.h>
#include "../utils/log_utils.h"
#include "../utils/tensor.h"

/**
 * @brief Converts 3D bounding boxes from world coordinates to camera coordinates
 *
 * Projects 3D bounding boxes onto 2D camera frames.
 *
 * @param bboxes3d Torch tensor containing 3D bounding boxes in world coordinates
 * @param projection_matrices Projection matrices for transforming from world to camera coordinates
 * @return Torch tensor of projected 2D coordinates for each corner of each box
 */
torch::Tensor generate_bbox3d_on_camera_frames(
    const torch::Tensor& bboxes3d,
    const torch::Tensor& projection_matrices);

/**
* @brief Generate 2D bounding boxes from projected 3D rectangle corners
*
* This function creates 2D bounding boxes from projected 3D corners using
* a simple and efficient min/max approach. It includes visibility checking
* to filter out objects that are not sufficiently visible in the frame.
*
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
    int image_height);

/**
 * @brief Converts 3D bounding box parameters to corner points in 3D space
 *
 * @param box3d Torch tensor containing 3D bounding box parameters
 * @return Torch tensor containing corner points for each bounding box
 */
torch::Tensor box3d_to_corners(const torch::Tensor& box3d);

/**
 * @brief Test function to generate bbox3d on camera frames
 *
 * @param logger Logger for debug and warning messages
 */
void test_generate_bbox3d_on_camera_frames(PrettyLogger* logger);
#endif