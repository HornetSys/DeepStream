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

#include "display_bbox3d.h"
#include <fstream>
#include <sstream>

// Constants for display
#define MAX_DISPLAY_LEN 64
#define NUM_CORNERS 8

// Helper function to trim whitespace efficiently
static std::string trim_whitespace(const std::string& str) {
    const std::string whitespace = " \t\n\r";
    size_t start = str.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        return "";  // String is all whitespace
    }
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

// Helper function to parse a single token
static bool parse_class_threshold_token(const std::string& token, 
                                       ClassThreshold& result, 
                                       PrettyLogger* logger) {
    size_t colon_pos = token.find(':');
    if (colon_pos == std::string::npos) {
        logger->warning("Invalid format for token '" + token + 
                      "', expected 'class_name:threshold', skipping entry");
        return false;
    }
    
    std::string class_name = trim_whitespace(token.substr(0, colon_pos));
    std::string threshold_str = trim_whitespace(token.substr(colon_pos + 1));
    
    if (class_name.empty()) {
        logger->warning("Empty class name in token '" + token + "', skipping entry");
        return false;
    }
    
    if (threshold_str.empty()) {
        logger->warning("Empty threshold in token '" + token + "', skipping entry");
        return false;
    }
    
    try {
        float threshold = std::stof(threshold_str);
        if (threshold < 0.0f || threshold > 1.0f) {
            logger->warning("Threshold value " + threshold_str + " for class '" + 
                          class_name + "' is out of range [0,1], using clamped value");
            threshold = std::max(0.0f, std::min(1.0f, threshold));
        }
        
        result.label = std::move(class_name);
        result.threshold = threshold;
        return true;
    } catch (const std::exception& e) {
        logger->warning("Invalid threshold value '" + threshold_str + 
                      "' for class '" + class_name + "': " + e.what() + ", skipping entry");
        return false;
    }
}

std::vector<ClassThreshold> fetch_class_labels_with_thresholds(
    const std::string& labels_file,
    PrettyLogger* logger
) {
    std::vector<ClassThreshold> class_thresholds;

    // Check if labels file path is valid
    if (labels_file.empty()) {
        logger->error("Labels file path is empty");
        return class_thresholds;
    }

    // Open the labels file
    std::ifstream file(labels_file);
    if (!file.is_open()) {
        logger->error("Error: Could not open labels file: " + labels_file + " (file may not exist)");
        return class_thresholds;
    }

    // Check if file is readable
    if (!file.good()) {
        logger->error("Error: Labels file is not readable: " + labels_file);
        file.close();
        return class_thresholds;
    }

    // Read entire file content
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();

    // Reserve space for better performance (estimate based on content size)
    class_thresholds.reserve(std::count(content.begin(), content.end(), ';') + 1);

    // Parse file content efficiently using iterators
    size_t start = 0;
    size_t pos = 0;
    
    while ((pos = content.find(';', start)) != std::string::npos) {
        if (pos > start) {  // Skip empty tokens
            std::string token = trim_whitespace(content.substr(start, pos - start));
            if (!token.empty()) {
                ClassThreshold class_threshold;
                if (parse_class_threshold_token(token, class_threshold, logger)) {
                    class_thresholds.push_back(std::move(class_threshold));
                }
            }
        }
        start = pos + 1;
    }

    // Handle the last token if it doesn't end with a semicolon
    if (start < content.length()) {
        std::string token = trim_whitespace(content.substr(start));
        if (!token.empty()) {
            ClassThreshold class_threshold;
            if (parse_class_threshold_token(token, class_threshold, logger)) {
                class_thresholds.push_back(std::move(class_threshold));
            }
        }
    }

    // Build debug string efficiently
    std::string class_thresholds_str;
    class_thresholds_str.reserve(class_thresholds.size() * 20);  // Estimate
    for (const auto& ct : class_thresholds) {
        class_thresholds_str += ct.label + ":" + std::to_string(ct.threshold) + ";";
    }

    // Log the number of classes found
    logger->info("Loaded " + std::to_string(class_thresholds.size()) + 
                " class labels with thresholds [" + class_thresholds_str + 
                "] from " + labels_file);

    return class_thresholds;
}

std::vector<std::string> extract_class_labels(
    const std::vector<ClassThreshold>& class_thresholds
) {
    std::vector<std::string> class_labels;
    class_labels.reserve(class_thresholds.size());
    
    for (const auto& ct : class_thresholds) {
        class_labels.push_back(ct.label);
    }
    
    return class_labels;
}

void clamp_point(float* x, float* y, int img_width, int img_height) {
    if (*x < 0) *x = 0;
    if (*x >= img_width) *x = img_width - 1;
    if (*y < 0) *y = 0;
    if (*y >= img_height) *y = img_height - 1;
}

void calculate_centroid(const float* points, float* centroid) {
    centroid[0] = 0;
    centroid[1] = 0;

    for (int j = 0; j < NUM_CORNERS; ++j) {
        centroid[0] += points[j * 2];      // x-coordinate
        centroid[1] += points[j * 2 + 1];  // y-coordinate
    }

    centroid[0] /= NUM_CORNERS;
    centroid[1] /= NUM_CORNERS;
}

void get_color_from_label(const std::string& label, float* red, float* green, float* blue, int num_classes) {
    // If label is empty, return a default color
    if (label.empty()) {
        *red = 0.7f;
        *green = 0.7f;
        *blue = 0.7f;
        return;
    }

    // Simple hash function for consistent color
    unsigned int hash = 0;
    for (char c : label) {
        hash = hash * 31 + c;
    }

    // Golden ratio approach for color distribution
    float hue = (hash % num_classes) / (float)num_classes;

    // Convert HSV to RGB (simplified, assuming S=1, V=1)
    float h = hue * 6.0f;
    int i = (int)h;
    float f = h - i;
    float p = 0.0f;
    float q = 1.0f - f;
    float t = f;

    switch (i % 6) {
        case 0: *red = 1.0f; *green = t; *blue = p; break;
        case 1: *red = q; *green = 1.0f; *blue = p; break;
        case 2: *red = p; *green = 1.0f; *blue = t; break;
        case 3: *red = p; *green = q; *blue = 1.0f; break;
        case 4: *red = t; *green = p; *blue = 1.0f; break;
        case 5: *red = 1.0f; *green = p; *blue = q; break;
    }
}

void get_color_from_instance_id(int64_t instance_id, float* red, float* green, float* blue) {
    // Special case for ID 0
    if (instance_id == 0) {
        *red = 0.0f;
        *green = 1.0f;
        *blue = 0.0f;
        return;
    }

    // Generate colors based on instanceId using modulo operation
    *red = (instance_id * 1234567) % 256 / 255.0f;
    *green = (instance_id * 2345678) % 256 / 255.0f;
    *blue = (instance_id * 3456789) % 256 / 255.0f;
}

bool count_visible_corners(const float* points, int num_corners, int img_width, int img_height) {
    int visible_corners = 0;
    for (int i = 0; i < num_corners; i++) {
        float x = points[i * 2];
        float y = points[i * 2 + 1];
        if (x >= 0 && x < img_width && y >= 0 && y < img_height) {
            visible_corners++;
        }
    }
    return visible_corners >= (num_corners / 2);
}

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
) {
    if (!batch_meta || !frame_meta) {
        logger->warning("Invalid metadata provided to attach_bbox3d_to_display");
        return;
    }

    // Define line indices for the 3D bounding box edges
    const int line_indices[][2] = {
        {0, 1}, {0, 3}, {0, 4}, {1, 2}, {1, 5},
        {3, 2}, {3, 7}, {4, 5}, {4, 7}, {2, 6},
        {5, 6}, {6, 7}
    };

    // Access the camera frames tensor data
    auto camera_frames_accessor = camera_frames.accessor<float, 3>();
    const int num_boxes = camera_frames.size(0);

    // Process each detected object
    for (int box_idx = 0; box_idx < num_boxes; ++box_idx) {
        // Extract the 2D points for this box
        float points[NUM_CORNERS * 2];
        for (int corner = 0; corner < NUM_CORNERS; corner++) {
            points[corner * 2] = camera_frames_accessor[box_idx][corner][0];
            points[corner * 2 + 1] = camera_frames_accessor[box_idx][corner][1];
        }

        // Skip if not enough corners are visible
        if (!count_visible_corners(points, NUM_CORNERS, img_width, img_height)) {
            continue;
        }

        // Acquire display metadata from the pool
        NvDsDisplayMeta* display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        if (!display_meta) {
            logger->warning("Failed to acquire display metadata from pool");
            continue;
        }

        // Get class label index and instance ID for this object
        int64_t label_idx = 0;
        int64_t instance_id = 0;

        if (labels_3d.defined() && box_idx < labels_3d.size(0)) {
            label_idx = labels_3d[box_idx].item<int64_t>();
        }

        if (instance_ids.defined() && box_idx < instance_ids.size(0)) {
            instance_id = instance_ids[box_idx].item<int64_t>();
        }

        // Get actual label string from the index
        std::string label = "";
        if (label_idx >= 0 && label_idx < static_cast<int64_t>(class_labels.size())) {
            label = class_labels[label_idx];
        }


        // Variables for color selection
        float red = 0.0f, green = 0.0f, blue = 0.0f;
        bool color_by_class = (bbox_color_mode == "class_id");

        if (color_by_class) {
            // Color by class label
            get_color_from_label(label, &red, &green, &blue, class_labels.size());
        } else {
            // Color by instance ID
            get_color_from_instance_id(instance_id, &red, &green, &blue);
        }

        // Draw all the lines of the 3D bounding box
        for (size_t edge = 0; edge < sizeof(line_indices) / sizeof(line_indices[0]); ++edge) {
            int start_corner = line_indices[edge][0];
            int end_corner = line_indices[edge][1];

            float x1 = points[start_corner * 2];
            float y1 = points[start_corner * 2 + 1];
            float x2 = points[end_corner * 2];
            float y2 = points[end_corner * 2 + 1];

            // Skip if both points are outside the image
            if ((x1 < 0 || x1 > img_width || y1 < 0 || y1 > img_height) &&
                (x2 < 0 || x2 > img_width || y2 < 0 || y2 > img_height)) {
                continue;
            }

            // Clamp points to image boundaries
            clamp_point(&x1, &y1, img_width, img_height);
            clamp_point(&x2, &y2, img_width, img_height);

            // Add line to display metadata
            if (display_meta->num_lines < MAX_ELEMENTS_IN_DISPLAY_META) {
                NvOSD_LineParams* line_params = &display_meta->line_params[display_meta->num_lines];
                line_params->x1 = x1;
                line_params->y1 = y1;
                line_params->x2 = x2;
                line_params->y2 = y2;
                line_params->line_width = 3;
                line_params->line_color.red = red;
                line_params->line_color.green = green;
                line_params->line_color.blue = blue;
                line_params->line_color.alpha = 1.0f;
                display_meta->num_lines++;
            }
        }

        // Add label text near centroid if we have a valid label or instance ID
        if (!label.empty() || instance_id != 0) {
            // Calculate centroid to place text
            float centroid[2];
            calculate_centroid(points, centroid);

            // Skip text if centroid is outside the image
            if (centroid[0] < 10 || centroid[0] >= img_width - 10 ||
                centroid[1] < 10 || centroid[1] >= img_height - 10) {
                continue;
            }

            // Check if we can add a text label
            if (display_meta->num_labels < MAX_ELEMENTS_IN_DISPLAY_META) {
                std::string display_text = "";

                // Add class label if enabled and label is not empty
                if (display_class_labels && !label.empty()) {
                    display_text += label + " ";
                }

                // Add instance ID if enabled
                if (display_instance_ids) {
                    display_text += std::to_string(instance_id) + " ";
                }

                // Add score if enabled and score tensor is available
                if (display_class_score && box_idx < scores_3d.size(0)) {
                    // Format the score with 2 decimal places
                    float score = scores_3d[box_idx].item<float>();
                    char score_str[16];
                    snprintf(score_str, sizeof(score_str), "[%.2f]", score);
                    display_text += score_str;
                }

                if (!display_text.empty()) {
                    NvOSD_TextParams* txt_params = &display_meta->text_params[display_meta->num_labels];
                    txt_params->display_text = g_strdup(display_text.c_str());
                    txt_params->x_offset = centroid[0] - 10;
                    txt_params->y_offset = centroid[1] - 10;
                    txt_params->font_params.font_name = const_cast<char*>("Arial");
                    txt_params->font_params.font_size = 13;
                    txt_params->font_params.font_color.red = 1.0f;
                    txt_params->font_params.font_color.green = 1.0f;
                    txt_params->font_params.font_color.blue = 1.0f;
                    txt_params->font_params.font_color.alpha = 1.0f;
                    txt_params->set_bg_clr = 0;
                    display_meta->num_labels++;
                }
            }
        }

        // Add display metadata to frame
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }
}

/**
 * @brief Attach 2D bounding boxes as NvDsObjectMeta to frame metadata
 *
 * This function creates NvDsObjectMeta for each 2D bounding box and attaches
 * them to the frame metadata. This allows the bounding boxes to be processed
 * by downstream elements in the DeepStream pipeline. Supports color modes for
 * consistent visualization based on class or instance.
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
) {
    if (!bbox2d.defined() || bbox2d.numel() == 0) {
        logger->warning("No 2D bounding boxes to attach as object metadata");
        return;
    }

    // Ensure tensors are on CPU
    torch::Tensor bbox2d_cpu = bbox2d.cpu();
    torch::Tensor labels_cpu = labels_3d.cpu();
    torch::Tensor instance_ids_cpu = instance_ids.cpu();
    torch::Tensor scores_cpu = scores_3d.cpu();

    // Get number of objects
    int64_t num_objects = bbox2d_cpu.size(0);

    // Access raw data
    float* bbox_data = bbox2d_cpu.data_ptr<float>();
    int64_t* label_data = labels_cpu.data_ptr<int64_t>();
    int64_t* instance_data = instance_ids_cpu.data_ptr<int64_t>();
    float* score_data = scores_cpu.data_ptr<float>();

    // Iterate through each object
    for (int64_t i = 0; i < num_objects; i++) {
        // Extract bbox coordinates [x, y, w, h]
        float x = bbox_data[i * 4 + 0];
        float y = bbox_data[i * 4 + 1];
        float width = bbox_data[i * 4 + 2];
        float height = bbox_data[i * 4 + 3];

        // Skip invalid bounding boxes (negative dimensions indicate non-visible objects)
        if (width <= 0 || height <= 0) {
            // This object was filtered out due to insufficient visibility
            continue;
        }

        // Clamp coordinates to frame boundaries
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + width > frame_meta->source_frame_width) {
            width = frame_meta->source_frame_width - x;
        }
        if (y + height > frame_meta->source_frame_height) {
            height = frame_meta->source_frame_height - y;
        }

        // Acquire object metadata from pool
        NvDsObjectMeta* obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
        if (!obj_meta) {
            logger->warning("Failed to acquire object metadata from pool");
            break;
        }

        // Set bounding box coordinates
        obj_meta->rect_params.left = x;
        obj_meta->rect_params.top = y;
        obj_meta->rect_params.width = width;
        obj_meta->rect_params.height = height;

        // Set class ID and confidence score
        obj_meta->class_id = static_cast<gint>(label_data[i]);
        obj_meta->confidence = score_data[i];

        // Set object ID from instance ID (for tracking)
        obj_meta->object_id = static_cast<guint64>(instance_data[i]);

        // Set class label if available
        if (obj_meta->class_id >= 0 &&
            obj_meta->class_id < static_cast<gint>(class_labels.size())) {
            g_strlcpy(obj_meta->obj_label,
                     class_labels[obj_meta->class_id].c_str(),
                     MAX_LABEL_SIZE);
        }

        // Set unique component ID to identify this as sparse4d detection
        // Using a non-zero ID to differentiate from nvinfer detections
        obj_meta->unique_component_id = 100;  // Sparse4D component ID

        if (display_2d_bbox) {
            // Set border appearance
            obj_meta->rect_params.border_width = 3;  // Thicker border for visibility
            obj_meta->rect_params.has_bg_color = 0;

            // Set color based on bbox_color_mode
            float red, green, blue;
            bool color_by_class = (bbox_color_mode == "class_id");

            if (color_by_class) {
                // Color by class label
                if (obj_meta->class_id >= 0 &&
                    obj_meta->class_id < static_cast<gint>(class_labels.size())) {
                    // Create a unique string for each class ID to ensure different colors
                    // Append class ID to label to make it unique even if labels are same
                    std::string unique_label = class_labels[obj_meta->class_id] + "_" + std::to_string(obj_meta->class_id);
                    get_color_from_label(unique_label,
                                    &red, &green, &blue,
                                    class_labels.size());
                } else {
                    // Default color if class is unknown
                    red = 1.0f;
                    green = 0.0f;
                    blue = 0.0f;
                }
            } else {
                // Color by instance ID - ensures same instance has same color across frames
                // This is useful for tracking where the same object should maintain its color
                get_color_from_instance_id(static_cast<int64_t>(obj_meta->object_id),
                                        &red, &green, &blue);
            }

            // Set rectangle parameters with explicit color
            obj_meta->rect_params.border_color.red = red;
            obj_meta->rect_params.border_color.green = green;
            obj_meta->rect_params.border_color.blue = blue;
            obj_meta->rect_params.border_color.alpha = 1.0f;

            // Add text display parameters
            obj_meta->text_params.display_text = g_strdup(obj_meta->obj_label);
            obj_meta->text_params.x_offset = x;
            obj_meta->text_params.y_offset = y - 10;
            obj_meta->text_params.set_bg_clr = 1;
            obj_meta->text_params.text_bg_clr = {0, 0, 0, 1};
            obj_meta->text_params.font_params.font_name = g_strdup("Serif");
            obj_meta->text_params.font_params.font_size = 10;
            obj_meta->text_params.font_params.font_color = {1, 1, 1, 1};
        } else {
            // display_2d_bbox is false, so hide the bounding box and text
            obj_meta->rect_params.border_width = 0;
            obj_meta->rect_params.has_bg_color = 0;
            obj_meta->text_params.font_params.font_size = 0;
            obj_meta->obj_label[0] = '\0';
        }

        // Attach object metadata to frame
        nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
    }
}