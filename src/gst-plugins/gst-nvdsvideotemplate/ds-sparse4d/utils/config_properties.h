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

#ifndef CONFIG_PROPERTIES_H
#define CONFIG_PROPERTIES_H

/**
 * @brief Structure containing all configurable properties from the YAML config file.
 *
 * This structure serves as a central repository for all configuration parameters
 * that can be set via the YAML config file. When the ConfigParser loads and parses
 * a configuration file, it populates this structure with the values from the file,
 * or uses the default values specified here if not present in the file.
 *
 * Components can access these properties directly through the ConfigParser's
 * getProperties() method instead of making individual calls to getInt(), getBool(), etc.
 */
struct ConfigProperties {
    // Inference properties
    std::string onnx_file = "";      // Path to the ONNX model file
    std::string engine_file = "";    // Path to save/load the TensorRT engine file
    std::string labels_file = "";    // Path to the labels file
    int batch_size = 1;              // Sparse4D model batch size
    int num_sensors = 6;             // Number of sensors [deepstream batch size]
    bool enable_fp16 = false;        // Whether to enable FP16 precision
    bool force_engine_rebuild = false; // Whether to force rebuild the engine file even if it exists
    bool feedback = true;            // Feedback flag
    bool partial_batch = true;       // Whether to use partial batch for inference
    int interval = 0;                // Number of consecutive batches to be skipped for inference

    int gpu_id = 0;  // Default to GPU 0 if not specified
    int preprocessed_height = 512;   // Height of preprocessed images
    int preprocessed_width = 1408;   // Width of preprocessed images

    // Anchor properties
    int num_anchor = 900;            // Number of anchors
    int embed_dims = 256;            // Dimension for embedding tensors
    std::string anchor = "";         // Path to anchor file
    std::string anchor_handler = ""; // Anchor handler name
    int num_temp_instances = 600;    // Number of temporal instances
    float default_time_interval = 0.05f; // Default time interval for temporal anchor projection
    float confidence_decay = 0.8f;   // Confidence decay rate
    bool anchor_grad = false;        // Anchor gradient flag
    bool feat_grad = false;          // Feature gradient flag
    int max_time_interval = 2;       // Maximum time interval for temporal tracking
    int reid_dims = -1;              // Dimensions for re-identification features


    // Decoder properties
    int num_output = 600;              // Number of outputs
    float score_threshold = 0.05f;     // Score threshold for decoding output

    // Performance properties
    int num_torch_threads = 8;         // Number of threads for PyTorch

    // Visualization properties
    bool generate_3d_bbox = true;    // Generate 3D bounding boxes for display
    bool generate_2d_bbox = false;      // Enable OCR inference
    std::string bbox_color_mode = "class_id"; // Color mode for bounding boxes (class_id or instance_id)
    bool display_class_labels = true;  // Whether to display class labels
    bool display_instance_ids = true;  // Whether to display instance IDs
    bool display_class_score = true;   // Whether to display class confidence score
    bool display_3d_bbox = false;      // Whether to display 3D bounding boxes on frames
    bool display_2d_bbox = false;       // Whether to display 2D bounding boxes on frames

    // Calibration properties
    std::string calib_file_path = "";        // Path to calibration file
    std::string bev_group_name = "";         // BEV group name
    std::string calib_mode = "synthetic";    // Calibration mode (synthetic, aic24, etc.)
    bool use_camera_groups = true;           // Whether to group cameras for multi-view fusion
    bool recentering = true;                 // Whether to recalculate camera coordinates from a reference point

    // Augmentation configurations
    struct AugmentationConfig {
        float resize = 1.0f;                   // Resize factor for images
        std::vector<int> resize_dims; // Dimensions after resizing [width, height]
        std::vector<int> crop; // Crop parameters [x, y, width, height]
        bool flip = false;                     // Whether to flip the image horizontally
        float rotate = 0.0f;                   // Rotation angle in degrees
        float rotate_3d = 0.0f;                // 3D rotation angle in degrees
    };

    AugmentationConfig aug_config;            // Augmentation configuration

    // Debugging properties
    std::string log_level = "error"; // Logging level: info, warn, error (persistent_info always displayed)
    bool display_partial_batch_info = true; // Whether to display partial batch information
    bool display_tensor_info = false; // Whether to display tensor information valid if silent is false
    bool display_bbox_list_info = false; // Whether to display bboxlist information
    int max_objects_to_display = 5;    // Maximum number of objects to display in bboxlist
    bool dump_frames = false;        // Whether to dump frames for debugging
    int dump_max_frames = 3;         // Number of frames to dump for debugging purposes
    std::string fixed_input_tensor_path = "";   // Path to fixed input tensor file
    bool gpu_postprocess = true;    // Whether to perform postprocessing (instance bank & decoder) on GPU
    bool use_static_frame_num = true;    // Whether to use plugin's own frame counter instead of parsed frame_num from batch meta

    // TensorRT properties
    int trt_logger_severity = 2;     // TensorRT logger severity level:
    bool enable_profiler = false;    // Whether to enable TensorRT layer profiling

    // Add more properties here as needed
};

#endif /* CONFIG_PROPERTIES_H */