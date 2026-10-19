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

#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <string>
#include <unordered_map>
#include <memory>
#include <yaml-cpp/yaml.h>
#include "config_properties.h"
#include <iostream>
#include <iomanip>

class ConfigParser {
public:
    // Constructor with default config path
    ConfigParser(const std::string& configPath = "./config.yaml");

    // Load a different config file
    bool loadConfig(const std::string& configPath);

    // Get string value from config
    std::string getString(const std::string& key, const std::string& defaultValue = "") const;

    // Get integer value from config
    int getInt(const std::string& key, int defaultValue = 0) const;

    // Get float value from config
    float getFloat(const std::string& key, float defaultValue = 0.0f) const;

    // Get boolean value from config
    bool getBool(const std::string& key, bool defaultValue = false) const;

    // Check if config has a key
    bool hasKey(const std::string& key) const;

    // Get the loaded config file path
    std::string getConfigPath() const;

    // Check if config was loaded successfully
    bool isLoaded() const;

    // Get the populated config properties
    const ConfigProperties& getProperties() const;

    // Add this method to the ConfigParser class
    void displayProperties() const {
        if (!m_IsLoaded) {
            std::cout << "No configuration loaded." << std::endl;
            return;
        }

        const int prop_width = 35;
        const int val_width = 80;
        const int table_width = prop_width + val_width + 5; // 5 for borders and separators

        // Helper function to create separator line
        auto separator = [prop_width, val_width]() {
            std::string sep = "+ ";
            sep += std::string(prop_width, '-');
            sep += " + ";
            sep += std::string(val_width, '-');
            sep += " +";
            return sep;
        };

        // Print table header
        std::cout << separator() << std::endl;
        std::cout << "│ ";
        std::cout << std::setw(table_width - 2) << std::left << "Configuration Properties (" + m_ConfigPath + ")";
        std::cout << " │" << std::endl;
        std::cout << separator() << std::endl;

        // Print column headers
        std::cout << "│ ";
        std::cout << std::setw(prop_width) << std::left << "Property";
        std::cout << " │ ";
        std::cout << std::setw(val_width) << std::left << "Value";
        std::cout << " │" << std::endl;

        // Inference properties
        std::cout << separator() << std::endl;
        std::cout << "│ " << std::setw(prop_width) << std::left << "onnx_file"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.onnx_file
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "engine_file"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.engine_file
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "labels_file"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.labels_file
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "gpu_id"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.gpu_id
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "batch_size"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.batch_size
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "num_sensors"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.num_sensors
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "enable_fp16"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.enable_fp16 ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "force_engine_rebuild"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.force_engine_rebuild ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "partial_batch"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.partial_batch ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "feedback"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.feedback ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "interval"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.interval
                  << " │" << std::endl;

        // Calibration properties
        std::cout << separator() << std::endl;
        std::cout << "│ " << std::setw(prop_width) << std::left << "calib_file_path"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.calib_file_path
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "bev_group_name"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.bev_group_name
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "calib_mode"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.calib_mode
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "use_camera_groups"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.use_camera_groups ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "recentering"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.recentering ? "true" : "false")
                  << " │" << std::endl;

        // Preprocess properties
        std::cout << separator() << std::endl;
        std::cout << "│ " << std::setw(prop_width) << std::left << "preprocessed_height"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.preprocessed_height
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "preprocessed_width"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.preprocessed_width
                  << " │" << std::endl;

        auto vector_to_string = [](const std::vector<int>& vec) {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < vec.size(); ++i) {
                ss << vec[i];
                if (i < vec.size() - 1) ss << ", ";
            }
            ss << "]";
            return ss.str();
        };

        std::cout << "│ " << std::setw(prop_width) << std::left << "aug_config.resize"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.aug_config.resize
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "aug_config.resize_dims"
                  << " │ " << std::setw(val_width) << std::left << vector_to_string(m_Properties.aug_config.resize_dims)
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "aug_config.crop"
                  << " │ " << std::setw(val_width) << std::left << vector_to_string(m_Properties.aug_config.crop)
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "aug_config.flip"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.aug_config.flip ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "aug_config.rotate"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.aug_config.rotate
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "aug_config.rotate_3d"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.aug_config.rotate_3d
                  << " │" << std::endl;


        // Postprocess properties
        std::cout << "│ " << std::setw(prop_width) << std::left << "num_anchor"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.num_anchor
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "embed_dims"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.embed_dims
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "anchor"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.anchor
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "anchor_handler"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.anchor_handler
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "num_temp_instances"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.num_temp_instances
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "default_time_interval"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.default_time_interval
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "confidence_decay"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.confidence_decay
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "anchor_grad"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.anchor_grad ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "feat_grad"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.feat_grad ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "max_time_interval"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.max_time_interval
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "reid_dims"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.reid_dims
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "num_output"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.num_output
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "score_threshold"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.score_threshold
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "num_torch_threads"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.num_torch_threads
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "generate_3d_bbox"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.generate_3d_bbox ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "generate_2d_bbox"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.generate_2d_bbox ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "bbox_color_mode"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.bbox_color_mode
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "display_class_labels"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.display_class_labels ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "display_instance_ids"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.display_instance_ids ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "display_class_score"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.display_class_score ? "true" : "false")
                  << " │" << std::endl;
        std::cout << "│ " << std::setw(prop_width) << std::left << "display_3d_bbox"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.display_3d_bbox ? "true" : "false")
                  << " │" << std::endl;
        std::cout << "│ " << std::setw(prop_width) << std::left << "display_2d_bbox"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.display_2d_bbox ? "true" : "false")
                  << " │" << std::endl;

        // Debugging properties
        std::cout << separator() << std::endl;
        std::cout << "│ " << std::setw(prop_width) << std::left << "log_level"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.log_level
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "display_partial_batch_info"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.display_partial_batch_info ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "display_tensor_info"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.display_tensor_info ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "display_bbox_list_info"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.display_bbox_list_info ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "max_objects_to_display"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.max_objects_to_display
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "dump_frames"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.dump_frames ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "dump_max_frames"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.dump_max_frames
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "fixed_input_tensor_path"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.fixed_input_tensor_path
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "gpu_postprocess"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.gpu_postprocess ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "use_static_frame_num"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.use_static_frame_num ? "true" : "false")
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "trt_logger_severity"
                  << " │ " << std::setw(val_width) << std::left << m_Properties.trt_logger_severity
                  << " │" << std::endl;

        std::cout << "│ " << std::setw(prop_width) << std::left << "enable_profiler"
                  << " │ " << std::setw(val_width) << std::left << (m_Properties.enable_profiler ? "true" : "false")
                  << " │" << std::endl;

        // Print table footer
        std::cout << separator() << std::endl;
    }

private:
    // Update properties from the loaded config
    void updatePropertiesFromConfig();

    std::string m_ConfigPath;
    bool m_IsLoaded;
    YAML::Node m_Config;
    ConfigProperties m_Properties;
};

#endif // CONFIG_PARSER_H