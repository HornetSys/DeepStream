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

#include "postprocessor.h"
#include <cstdlib>

// Static instance of the instance bank for reuse
static InstanceBank* s_instance_bank = nullptr;

// Static instance of the decoder for reuse
static SparseBox3DDecoder* s_decoder = nullptr;

// Static vectors to hold class labels and thresholds loaded once
static std::vector<ClassThreshold> s_ClassThresholds;
static std::vector<std::string> s_ClassLabels;

bool postprocess_tensors(TensorCache* tensorCache,
                         NvDsBatchMeta* batch_meta,
                         const ConfigProperties& config_properties,
                         PrettyLogger* logger) {

    // Set PyTorch thread limits to prevent excessive CPU usage
    static bool thread_limits_set = false;
    if (!thread_limits_set && config_properties.num_torch_threads > 0) {
        torch::set_num_threads(config_properties.num_torch_threads);
        torch::set_num_interop_threads(1);

        // Also set environment variables for any child processes
        setenv("OMP_NUM_THREADS", std::to_string(config_properties.num_torch_threads).c_str(), 1);
        setenv("MKL_NUM_THREADS", std::to_string(config_properties.num_torch_threads).c_str(), 1);

        logger->info("Set PyTorch thread limits: num_threads=" + std::to_string(config_properties.num_torch_threads));
        thread_limits_set = true;
    }

    if (!batch_meta) {
        logger->warning("No batch metadata provided, detection results cannot be attached to frame");
    }

    int frame_num = tensorCache->getTensorFrameNum("img");
    int gpu_id = tensorCache->getTensorGpuId("img");
    gpu_id = config_properties.gpu_postprocess ? gpu_id : -1;
    std::vector<torch::Tensor> classification(6);
    std::vector<torch::Tensor> prediction(6);
    std::vector<torch::Tensor> quality(6);
    torch::Tensor output_cached_feature;
    torch::Tensor output_cached_anchor;
    torch::Tensor instance_id = torch::Tensor();

    // Lock metadata before modifications
    nvds_acquire_meta_lock(batch_meta);
    NvDsMetaList* l_frame;
    NvDsFrameMeta *frame_meta;

    // Get the class labels and thresholds only once and reuse
    if (s_ClassLabels.empty()) {
        try {
            s_ClassThresholds = fetch_class_labels_with_thresholds(config_properties.labels_file, logger);
            s_ClassLabels = extract_class_labels(s_ClassThresholds);
            
            if (s_ClassLabels.empty()) {
                logger->warning("No class labels loaded from " + config_properties.labels_file + ", using empty label set");
            }
        } catch (const std::exception& e) {
            logger->error("Error loading class labels: " + std::string(e.what()) + ", using empty label set");
            s_ClassThresholds.clear();
            s_ClassLabels.clear();
        }
    }

    // Calculate mean timestamp
    guint64 mean_timestamp = calculate_mean_timestamp(batch_meta);

    if (config_properties.gpu_postprocess) {
        logger->processStart("Reading Inference Output Tensors from GPU", frame_num);
        classification[0] = tensorCache->getTorchTensor("classification1");  // classification1
        classification[5] = tensorCache->getTorchTensor("classification2");  // classification2
        for (size_t i = 0; i < 6; i++)
            prediction[i] = tensorCache->getTorchTensor("prediction" + std::to_string(i + 1));  // prediction1-6
        quality[0] = tensorCache->getTorchTensor("quality1");  // quality1
        quality[5] = tensorCache->getTorchTensor("quality2");  // quality2
        output_cached_feature = tensorCache->getTorchTensor("output_cached_feature");  // output_cached_feature
        output_cached_anchor = tensorCache->getTorchTensor("output_cached_anchor");  // output_cached_anchor
        logger->processComplete("Reading Inference Output Tensors from GPU", frame_num, true);
    } else{
        logger->processStart("Cloning Inference Output Tensors to CPU", frame_num);
        classification[0] = tensorCache->cloneTorchTensorCPU("classification1");  // classification1
        classification[5] = tensorCache->cloneTorchTensorCPU("classification2");  // classification2
        for (size_t i = 0; i < 6; i++)
            prediction[i] = tensorCache->cloneTorchTensorCPU("prediction" + std::to_string(i + 1));  // prediction1-6
        quality[0] = tensorCache->cloneTorchTensorCPU("quality1");  // quality1
        quality[5] = tensorCache->cloneTorchTensorCPU("quality2");  // quality2
        output_cached_feature = tensorCache->cloneTorchTensorCPU("output_cached_feature");  // output_cached_feature
        output_cached_anchor = tensorCache->cloneTorchTensorCPU("output_cached_anchor");  // output_cached_anchor
        logger->processComplete("Cloning Inference Output Tensors to CPU", frame_num, true);
    }

    // Update the input tensors using the instance bank
    if (config_properties.feedback) {
        logger->processStart("Instance Bank", frame_num);
        AnchorHandler* anchor_handler = nullptr;

        // Create instance bank only once
        if (!s_instance_bank) {
            // Parse tensor form anchor file
            torch::Tensor anchor = load_anchor(config_properties.anchor);
            if (config_properties.anchor_handler == "SparseBox3DKeyPointsGenerator") {
                //create anchor handler
                anchor_handler = new SparseBox3DKeyPointsGenerator();
            } else {
                logger->warning("Unsupported anchor handler: " + config_properties.anchor_handler);
            }
            // Create instance bank
            s_instance_bank = new InstanceBank(
                config_properties.num_anchor,
                config_properties.embed_dims,
                anchor,
                anchor_handler,
                config_properties.num_temp_instances,
                config_properties.default_time_interval,
                config_properties.confidence_decay,
                config_properties.anchor_grad,
                config_properties.feat_grad,
                config_properties.max_time_interval,
                config_properties.reid_dims,
                gpu_id
            );
        }

        MetaData metas;
        torch::Device device = (gpu_id >= 0) ? torch::Device(torch::kCUDA, gpu_id) : torch::Device(torch::kCPU);
        metas["timestamp"] = torch::tensor(static_cast<int64_t>(mean_timestamp), torch::kInt64).to(device);

        if (frame_num > 0) {
            // Update instance bank
            logger->info("Updating Instance Bank");
            s_instance_bank->update(output_cached_feature, output_cached_anchor, classification[5]);
        } else {
            // Update prev_exists input tensor
            logger->info("Updating Prev Exists Tensor");
            intiPrevExistsTensor(tensorCache, 1.0);
        }

        // Cache instance bank
        logger->info("Caching Instance Bank");
        s_instance_bank->cache(output_cached_feature, output_cached_anchor, classification[5], &metas);

        // Get Instance ID
        logger->info("Getting Instance ID");
        // Use minimum threshold from all classes for instance tracking
        float min_threshold = 0.5f; // Default fallback
        if (!s_ClassThresholds.empty()) {
            min_threshold = s_ClassThresholds[0].threshold;
            for (const auto& ct : s_ClassThresholds) {
                if (ct.threshold < min_threshold) {
                    min_threshold = ct.threshold;
                }
            }
        }
        instance_id = s_instance_bank->get_instance_id(classification[5], output_cached_anchor, min_threshold);

        // Getting Feedback tensors from instance bank
        logger->info("Getting Feedback tensors from instance bank");
        torch::Tensor instance_feature;
        torch::Tensor anchor;
        torch::Tensor feedback_cached_feature;
        torch::Tensor feedback_cached_anchor;
        torch::Tensor time_interval;
        torch::Tensor reid_feature;
        std::tie(instance_feature, anchor, feedback_cached_feature, feedback_cached_anchor, time_interval, reid_feature) = s_instance_bank->get(config_properties.batch_size, metas);

        // Update interval_mask_tensor
        logger->info("Updating Feedback Tensor: Interval Mask Tensor");
        if (s_instance_bank->getMask().defined()) {
            initIntervalMaskTensor(tensorCache, s_instance_bank->getMask().item<float>());
        } else {
            initIntervalMaskTensor(tensorCache, 1.0f);
        }

        // Update input cached anchor with feedback_cached_anchor
        logger->info("Updating Feedback Tensor: Input Cached Anchor");
        if (feedback_cached_anchor.defined()) {
            updateInputCachedAnchorTensor(tensorCache, feedback_cached_anchor);
        }

        // Update input cached feature with feedback_cached_feature
        logger->info("Updating Feedback Tensor: Input Cached Feature");
        if (feedback_cached_feature.defined()) {
            updateInputCachedFeatureTensor(tensorCache, feedback_cached_feature);
        }

        logger->processComplete("Instance Bank", frame_num, true);
    }

    logger->processStart("Decoding Tensors", frame_num);
    // Create or update SparseBox3DDecoder instance only 1st time
    if (!s_decoder) {
        // Get group_origin tensor based on gpu_postprocess setting
        torch::Tensor group_origin;
        if (config_properties.gpu_postprocess) {
            group_origin = tensorCache->getTorchTensor("group_origin");
        } else {
            group_origin = tensorCache->cloneTorchTensorCPU("group_origin");
        }

        s_decoder = new SparseBox3DDecoder(
            config_properties.num_output,
            config_properties.score_threshold,
            true,
            group_origin,
            gpu_id,
            logger
        );
    }
    // Process torch tensors with decoder
    std::vector<DecodedResult> results;
    results = s_decoder->decode(
        classification,
        prediction,
        instance_id,
        output_cached_feature,
        quality
    );
   logger->processComplete("Decoding Tensors", frame_num, true);

    // Filter results based on class-specific confidence thresholds
    if (!s_ClassThresholds.empty()) {
        logger->processStart("Filtering Results with Class-specific Thresholds", frame_num);
        try {
            filter_results(results, s_ClassThresholds, logger);
        } catch (const std::exception& e) {
            logger->error("Error in class-specific filtering: " + std::string(e.what()));
        }
        logger->processComplete("Filtering Results with Class-specific Thresholds", frame_num, true);
    } else {
        logger->info("No class thresholds available, skipping filtering");
    }

    // if instance_ids is not defined, create a new instance_ids tensor with zero class ids
    if (!results.empty() && !results[0].instance_ids.defined() && results[0].boxes_3d.defined()) {
        auto device = results[0].boxes_3d.device();
        results[0].instance_ids = torch::zeros({results[0].boxes_3d.size(0)}, 
                                              torch::TensorOptions().dtype(torch::kInt64).device(device));
    }

    // Convert all tensors to CPU if GPU postprocessing is enabled
    if (!results.empty() && config_properties.gpu_postprocess) {
        logger->processStart("Converting Results Tensors to CPU", frame_num);
        // Check if each tensor is defined before converting to CPU
        results[0].instance_ids = results[0].instance_ids.to(torch::kCPU);
        results[0].boxes_3d = results[0].boxes_3d.to(torch::kCPU);
        results[0].scores_3d = results[0].scores_3d.to(torch::kCPU);
        results[0].cls_scores = results[0].cls_scores.to(torch::kCPU);
        results[0].labels_3d = results[0].labels_3d.to(torch::kCPU);
        results[0].instance_feats = results[0].instance_feats.defined() ?
                                    results[0].instance_feats.to(torch::kCPU) : results[0].instance_feats;
        logger->processComplete("Converting Results Tensors to CPU", frame_num, true);
    }

    // Display output tensor information
    if (config_properties.display_tensor_info && !results.empty()) {
        torch_tensor_display(results[0].boxes_3d, "boxes_3d", true);
        torch_tensor_display(results[0].scores_3d, "scores_3d", true);
        torch_tensor_display(results[0].cls_scores, "cls_scores", true);
        torch_tensor_display(results[0].labels_3d, "labels_3d", true);
        torch_tensor_display(results[0].instance_ids, "instance_ids", true);
        torch_tensor_display(results[0].instance_feats, "instance_feats", true);
    }

    if (config_properties.generate_3d_bbox && !results.empty() && results[0].boxes_3d.defined()) {
        logger->processStart("Generating BBox World Coordinates to Camera Coordinates", frame_num);
        // Get projection matrix based on gpu_postprocess setting
        torch::Tensor projection_matrix;
        if (config_properties.gpu_postprocess) {
            projection_matrix = tensorCache->getTorchTensor("org_projection_mat");
        } else {
            projection_matrix = tensorCache->cloneTorchTensorCPU("org_projection_mat");
        }
        // Get image dimensions from first frame in batch
        int img_width = 1920, img_height = 1080;
        if (batch_meta->frame_meta_list) {
            NvDsFrameMeta* first_frame = (NvDsFrameMeta*)(batch_meta->frame_meta_list->data);
            if (first_frame) {
                img_width = first_frame->source_frame_width;
                img_height = first_frame->source_frame_height;
            }
        }
        // Generate rect3d and bbox2d
        std::vector<torch::Tensor> batch_objects;
        std::vector<torch::Tensor> batch_bbox2d;
        for (size_t idx = 0; idx < results.size(); ++idx) {
            // Generate 3D bbox corners for each camera sensor
            torch::Tensor rect_3d = generate_bbox3d_on_camera_frames(
                results[idx].boxes_3d,
                projection_matrix[idx]
            );
            batch_objects.push_back(rect_3d);

            if (config_properties.generate_2d_bbox) {
                // Generate 2D bbox corners from 3D bbox corners
                torch::Tensor bbox2d = generate_bbox2d_from_rect3d(rect_3d, img_width, img_height);
                batch_bbox2d.push_back(bbox2d);
            }
        }
        if (config_properties.display_tensor_info) {
            torch_tensor_display(batch_objects[0], "rect_3d", true);
        }
        logger->processComplete("Generating BBox World Coordinates to Camera Coordinates", frame_num, true);

        logger->processStart("Attaching Detection Results to Display Meta", frame_num);
        // Attach detection results to frame
        for (size_t idx = 0; idx < results.size(); ++idx) {
            for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
            l_frame = l_frame->next) {
                frame_meta = (NvDsFrameMeta *) (l_frame->data);

                // Attach 3D bounding boxes to the frame using the utility function
                if (config_properties.display_3d_bbox) {
                    attach_bbox3d_to_display(
                    batch_meta,
                    frame_meta,
                    batch_objects[idx][frame_meta->batch_id],
                    results[idx].boxes_3d,
                    results[idx].labels_3d,
                    results[idx].instance_ids,
                    results[idx].scores_3d,
                    frame_meta->source_frame_width,
                    frame_meta->source_frame_height,
                    s_ClassLabels,
                    config_properties.bbox_color_mode,
                    config_properties.display_class_labels,
                    config_properties.display_instance_ids,
                    config_properties.display_class_score,
                    logger
                    );
                }

                // Attach 2D bounding boxes as object metadata to process by ocr library
                if (config_properties.generate_2d_bbox) {
                    attach_bbox2d_to_objectmeta(
                        batch_meta,
                        frame_meta,
                        batch_bbox2d[idx][frame_meta->batch_id],
                        results[idx].boxes_3d,
                        results[idx].labels_3d,
                        results[idx].instance_ids,
                        results[idx].scores_3d,
                        s_ClassLabels,
                        config_properties.bbox_color_mode,
                        config_properties.display_2d_bbox,
                        logger
                    );
                }
            }
        }
        logger->processComplete("Attaching Detection Results to Display Meta", frame_num, true);
    }

    // Attach bbox3dList to batch_meta as a user meta
    logger->processStart("Attaching bbox3dList to batch_meta as a user meta", frame_num);

    NvDsUserMeta* user_meta = NULL;
    NvDsBbox3dObjectList * bbox3dList = g_new0(NvDsBbox3dObjectList, 1);
    int num_detections = 0;
    if (!results.empty()) {
        num_detections = results[0].boxes_3d.size(0);
    }
    bbox3dList->count = 0;
    bbox3dList->num_objects = num_detections;
    bbox3dList->sensorStr = g_strdup(config_properties.bev_group_name.c_str());
    bbox3dList->timestamp = mean_timestamp;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
    l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta *) (l_frame->data);
        bbox3dList->camera_id = frame_meta->source_id;
        if (config_properties.use_static_frame_num) {
            bbox3dList->frame_num = frame_num;
        } else {
            bbox3dList->frame_num = frame_meta->frame_num;
        }
        break;
    }

    // Populate the bbox3dList with detection data
    if (num_detections > 0) {
        // Get pointers to the result tensor data
        auto boxes_3d_ptr = results[0].boxes_3d.data_ptr<float>();
        auto scores_3d_ptr = results[0].scores_3d.data_ptr<float>();
        auto cls_scores_ptr = results[0].cls_scores.data_ptr<float>();
        auto labels_3d_ptr = results[0].labels_3d.data_ptr<int64_t>();
        auto instance_ids_ptr = results[0].instance_ids.data_ptr<int64_t>();
        auto instance_feats_ptr = results[0].instance_feats.defined() ? 
                                 results[0].instance_feats.data_ptr<float>() : nullptr;

        int feat_dim = results[0].instance_feats.defined() ? results[0].instance_feats.size(1) : 0;

        // For each detection, populate the corresponding object in bbox3dList
        for (int i = 0; i < num_detections; i++) {
            NvDsBbox3dObject &obj = bbox3dList->objects[i];

            // Copy boxes_3d data (assuming 12 parameters per box)
            for (int j = 0; j < 12; j++) {
                obj.boxes_3d[j] = boxes_3d_ptr[i * 12 + j];
            }

            // Set scalar values
            obj.scores_3d = scores_3d_ptr[i];
            obj.cls_scores = cls_scores_ptr[i];

            // Set instance_ids
            obj.instance_ids = static_cast<guint>(instance_ids_ptr[i]);

            // Create class label string
            int64_t label_idx = labels_3d_ptr[i];
            if (label_idx >= 0 && label_idx < static_cast<int64_t>(s_ClassLabels.size())) {
                obj.labels_3d = g_strdup(s_ClassLabels[label_idx].c_str());
            } else {
                obj.labels_3d = g_strdup("unknown");
            }

            // Initialize OCR label as empty string
            obj.ocr_label = g_strdup("");

            // Handle instance features if available
            if (instance_feats_ptr != nullptr && feat_dim > 0) {
                obj.instance_feats = g_new0(NvDsEmbedding, 1);
                obj.instance_feats->embedding_length = feat_dim;
                obj.instance_feats->embedding_vector = g_new0(float, feat_dim);

                // Copy embedding vector
                for (int j = 0; j < feat_dim; j++) {
                    obj.instance_feats->embedding_vector[j] = instance_feats_ptr[i * feat_dim + j];
                }
            } else {
                obj.instance_feats = nullptr;
            }
        }
    }

    if (config_properties.display_bbox_list_info) {
        display_bbox3d_list(bbox3dList, config_properties);
    }

    user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    if (user_meta) {
        user_meta->user_meta_data = bbox3dList;
        user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_CUSTOM_MSG_SPARSE4D;
        user_meta->base_meta.copy_func =  (NvDsMetaCopyFunc) copy_user_meta;
        user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) release_user_meta_at_batch_level;
        user_meta->base_meta.batch_meta = batch_meta;
        nvds_add_user_meta_to_batch (batch_meta, user_meta);
    }

    logger->processComplete("Attaching bbox3dList to batch_meta as a user meta", frame_num, true);

    /* Release lock after modifications are complete */
    nvds_release_meta_lock(batch_meta);

    return true;
}


guint64 calculate_mean_timestamp(NvDsBatchMeta* batch_meta) {
    if (!batch_meta || batch_meta->num_frames_in_batch == 0) {
        return 0;
    }

    // Find minimum timestamp to use as base (to avoid overflow)
    NvDsMetaList* l_frame = batch_meta->frame_meta_list;
    NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)(l_frame->data);
    guint64 min_timestamp = frame_meta->ntp_timestamp;

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta*)(l_frame->data);
        if (frame_meta->ntp_timestamp < min_timestamp) {
            min_timestamp = frame_meta->ntp_timestamp;
        }
    }

    // Calculate sum of offsets from minimum (to prevent overflow)
    guint64 offset_sum = 0;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta*)(l_frame->data);
        offset_sum += (frame_meta->ntp_timestamp - min_timestamp);
    }

    // Calculate mean by adding minimum back to the mean offset
    guint64 mean_timestamp = min_timestamp + (offset_sum / batch_meta->num_frames_in_batch);
    return mean_timestamp;
}

gpointer copy_user_meta(gpointer data, gpointer user_data)
{
    if (!data) return nullptr;

    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    if (!user_meta || !user_meta->user_meta_data) return nullptr;

    NvDsBbox3dObjectList *src_bbox3dList = (NvDsBbox3dObjectList *)user_meta->user_meta_data;
    if (!src_bbox3dList || src_bbox3dList->num_objects > MAX_NUM_OBJECTS) return nullptr;

    // Allocate memory for the destination NvDsBbox3dObjectList
    NvDsBbox3dObjectList *dest_bbox3dList = g_new0(NvDsBbox3dObjectList, 1);

    // Copy simple fields from source to destination
    dest_bbox3dList->num_objects = src_bbox3dList->num_objects;
    dest_bbox3dList->timestamp = src_bbox3dList->timestamp;
    dest_bbox3dList->frame_num = src_bbox3dList->frame_num;
    dest_bbox3dList->camera_id = src_bbox3dList->camera_id;

    // Deep copy the sensor string if valid
    if (src_bbox3dList->sensorStr && g_utf8_validate(src_bbox3dList->sensorStr, -1, nullptr)) {
        dest_bbox3dList->sensorStr = g_strdup(src_bbox3dList->sensorStr);
    } else {
        dest_bbox3dList->sensorStr = nullptr;
    }

    // Loop through all the objects in the source list
    for (guint objIdx = 0; objIdx < src_bbox3dList->num_objects; ++objIdx) {
        NvDsBbox3dObject &src_bboxObj = src_bbox3dList->objects[objIdx];
        NvDsBbox3dObject &dest_bboxObj = dest_bbox3dList->objects[objIdx];

        // Copy boxes_3d array (static-sized)
        std::memcpy(dest_bboxObj.boxes_3d, src_bboxObj.boxes_3d, sizeof(src_bboxObj.boxes_3d));

        // Copy simple fields
        dest_bboxObj.scores_3d = src_bboxObj.scores_3d;
        dest_bboxObj.cls_scores = src_bboxObj.cls_scores;
        dest_bboxObj.instance_ids = src_bboxObj.instance_ids;

        // Deep copy the label string if valid
        if (src_bboxObj.labels_3d && g_utf8_validate(src_bboxObj.labels_3d, -1, nullptr)) {
            dest_bboxObj.labels_3d = g_strdup(src_bboxObj.labels_3d);
        } else {
            dest_bboxObj.labels_3d = nullptr;
        }

        // Deep copy the OCR label string if valid
        if (src_bboxObj.ocr_label && g_utf8_validate(src_bboxObj.ocr_label, -1, nullptr)) {
            dest_bboxObj.ocr_label = g_strdup(src_bboxObj.ocr_label);
        } else {
            dest_bboxObj.ocr_label = nullptr;
        }

        // Deep copy the instance_feats embedding vector if present
        if (src_bboxObj.instance_feats && src_bboxObj.instance_feats->embedding_length > 0) {
            dest_bboxObj.instance_feats = g_new0(NvDsEmbedding, 1);
            dest_bboxObj.instance_feats->embedding_length = src_bboxObj.instance_feats->embedding_length;

            // Allocate memory for embedding vector in destination
            dest_bboxObj.instance_feats->embedding_vector = g_new0(float, src_bboxObj.instance_feats->embedding_length);

            // Copy embedding vector from source to destination
            std::memcpy(dest_bboxObj.instance_feats->embedding_vector,
                        src_bboxObj.instance_feats->embedding_vector,
                        src_bboxObj.instance_feats->embedding_length * sizeof(float));
        } else {
            dest_bboxObj.instance_feats = nullptr;
        }
    }

    return (gpointer)dest_bbox3dList;
}

/**
 * @brief Display the contents of a 3D bounding box object list to console
 *
 * This function displays detailed information about a NvDsBbox3dObjectList to the
 * standard output, including all object attributes and features.
 *
 * @param bbox3dList Pointer to the NvDsBbox3dObjectList to display
 * @param config_properties Configuration properties containing display settings
 */
void display_bbox3d_list(const NvDsBbox3dObjectList* bbox3dList, const ConfigProperties& config_properties) {
    if (!bbox3dList) {
        std::cout << "bbox3dList is null" << std::endl;
        return;
    }

    // Table formatting constants
    const int lineWidth = 100;
    const std::string headerBorder = "+" + std::string(lineWidth+1, '-') + "+";
    const int labelWidth = 16;
    const int valueWidth = lineWidth - labelWidth - 3; // -3 for "| " and " |"
    const int max_items_to_display = 10;

    std::cout << "\n" << headerBorder << std::endl;

    // Create header with 3D bounding box list name
    std::string headerText = "3D Bounding Box List - " +
        (bbox3dList->sensorStr ? std::string(bbox3dList->sensorStr) : "Unnamed");
    int padding = (lineWidth - headerText.length()) / 2;
    std::cout << "| " << std::string(padding, ' ') << headerText
            << std::string(lineWidth - padding - headerText.length(), ' ') << "|" << std::endl;

    std::cout << headerBorder << std::endl;

    // Display general information with consistent formatting
    std::cout << "| Sensor          | " << std::left << std::setw(valueWidth)
              << (bbox3dList->sensorStr ? bbox3dList->sensorStr : "null") << " |" << std::endl;
    std::cout << "| Object Count    | " << std::left << std::setw(valueWidth)
              << bbox3dList->num_objects << " |" << std::endl;
    std::cout << "| Timestamp       | " << std::left << std::setw(valueWidth)
              << bbox3dList->timestamp << " |" << std::endl;
    std::cout << "| Camera ID       | " << std::left << std::setw(valueWidth)
              << bbox3dList->camera_id << " |" << std::endl;
    std::cout << "| Frame Number    | " << std::left << std::setw(valueWidth)
              << bbox3dList->frame_num << " |" << std::endl;
    std::cout << "| Count           | " << std::left << std::setw(valueWidth)
              << bbox3dList->count << " |" << std::endl;

    // Divider for objects section
    std::cout << "|" << std::string(lineWidth+1, '-') << "|" << std::endl;

    // Display details of objects with structured format
    int max_objects_to_display = config_properties.max_objects_to_display;
    for (int obj_idx = 0; obj_idx < std::min(max_objects_to_display, (int)bbox3dList->num_objects); obj_idx++) {
        // Object header
        std::string objHeader = "Object[" + std::to_string(obj_idx) + "]";
        std::cout << "| " << std::left << std::setw(labelWidth+3) << objHeader
                  << std::string(valueWidth - 1, ' ') << " |" << std::endl;

        std::cout << "| instance_ids    | " << std::left << std::setw(valueWidth)
                  << bbox3dList->objects[obj_idx].instance_ids << " |" << std::endl;
        std::string labels_str = bbox3dList->objects[obj_idx].labels_3d ?
            std::string(bbox3dList->objects[obj_idx].labels_3d) : "null";
        std::cout << "| labels_3d       | " << std::left << std::setw(valueWidth)
                  << labels_str << " |" << std::endl;
        std::string ocr_label_str = bbox3dList->objects[obj_idx].ocr_label ?
            std::string(bbox3dList->objects[obj_idx].ocr_label) : "null";
        std::cout << "| ocr_label       | " << std::left << std::setw(valueWidth)
                  << ocr_label_str << " |" << std::endl;
        std::cout << "| scores_3d       | " << std::left << std::fixed << std::setprecision(6) << std::setw(valueWidth)
                  << bbox3dList->objects[obj_idx].scores_3d << " |" << std::endl;
        std::cout << "| cls_scores      | " << std::left << std::fixed << std::setprecision(6) << std::setw(valueWidth)
                  << bbox3dList->objects[obj_idx].cls_scores << " |" << std::endl;

        // Format boxes_3d array as a string
        std::stringstream boxes_ss;
        boxes_ss << "[";
        for (int i = 0; i < 12; i++) {
            boxes_ss << std::fixed << std::setprecision(4) << bbox3dList->objects[obj_idx].boxes_3d[i];
            if (i < 11) boxes_ss << ", ";
        }
        boxes_ss << "]";

        // Wrap long boxes string if needed
        std::string boxes_str = boxes_ss.str();
        if (boxes_str.length() > valueWidth) {
            // Find a good breakpoint - try to break after a comma
            size_t breakPoint = valueWidth - 1;
            while (breakPoint > 0 && boxes_str[breakPoint] != ',') breakPoint--;

            if (breakPoint > 0) {
                // Split at comma
                std::string firstPart = boxes_str.substr(0, breakPoint + 1);
                std::string secondPart = boxes_str.substr(breakPoint + 1);

                // Trim leading spaces from secondPart
                secondPart.erase(0, secondPart.find_first_not_of(" "));

                // Display with proper alignment
                std::cout << "| boxes_3d        | " << std::left
                          << firstPart << std::string(valueWidth - firstPart.length(), ' ') << " |" << std::endl;
                std::cout << "|                 | " << std::left
                          << secondPart << std::string(valueWidth - secondPart.length(), ' ') << " |" << std::endl;
            } else {
                // If no good breakpoint found, just print full string
                std::cout << "| boxes_3d        | " << std::left << boxes_str
                          << std::string(valueWidth - boxes_str.length(), ' ') << " |" << std::endl;
            }
        } else {
            // String fits on one line
            std::cout << "| boxes_3d        | " << std::left << boxes_str
                      << std::string(valueWidth - boxes_str.length(), ' ') << " |" << std::endl;
        }

        // Handle instance_feats which is a pointer to NvDsEmbedding
        if (bbox3dList->objects[obj_idx].instance_feats) {
            std::stringstream feats_ss;
            feats_ss << "[";
            int feat_len = bbox3dList->objects[obj_idx].instance_feats->embedding_length;
            for (int i = 0; i < std::min(max_items_to_display, feat_len); i++) {
                feats_ss << std::fixed << std::setprecision(4) 
                         << bbox3dList->objects[obj_idx].instance_feats->embedding_vector[i];
                if (i < std::min(max_items_to_display, feat_len) - 1) feats_ss << ", ";
            }
            if (feat_len > max_items_to_display) feats_ss << ", ...";
            feats_ss << "]";

            // Handle wrapping for feature vector
            std::string feats_str = feats_ss.str();
            if (feats_str.length() > valueWidth) {
                // Find a good breakpoint - try to break after a comma
                size_t breakPoint = valueWidth - 1;
                while (breakPoint > 0 && feats_str[breakPoint] != ',') breakPoint--;

                if (breakPoint > 0) {
                    // Split at comma
                    std::string firstPart = feats_str.substr(0, breakPoint + 1);
                    std::string secondPart = feats_str.substr(breakPoint + 1);

                    // Trim leading spaces from secondPart
                    secondPart.erase(0, secondPart.find_first_not_of(" "));

                    // Display with proper alignment
                    std::cout << "| instance_feats  | " << std::left
                              << firstPart << std::string(valueWidth - firstPart.length(), ' ') << " |" << std::endl;
                    std::cout << "|                 | " << std::left
                              << secondPart << std::string(valueWidth - secondPart.length(), ' ') << " |" << std::endl;
                } else {
                    // If no good breakpoint found, just print full string
                    std::cout << "| instance_feats  | " << std::left << feats_str
                              << std::string(valueWidth - feats_str.length(), ' ') << " |" << std::endl;
                }
            } else {
                // String fits on one line
                std::cout << "| instance_feats  | " << std::left << feats_str
                          << std::string(valueWidth - feats_str.length(), ' ') << " |" << std::endl;
            }
        } else {
            std::cout << "| instance_feats  | " << std::left
                      << "null" << std::string(valueWidth - 4, ' ') << " |" << std::endl;
        }

        // Add divider between objects except after the last one
        if (obj_idx < std::min(max_objects_to_display, (int)bbox3dList->num_objects) - 1) {
            std::cout << "|" << std::string(lineWidth+1, '-') << "|" << std::endl;
        }
    }

    // If there are more objects than we displayed, indicate that
    if (max_objects_to_display < (int)bbox3dList->num_objects) {
        std::cout << "|" << std::string(lineWidth+1, ' ') << "|" << std::endl;
        std::cout << "| " << std::string(4, ' ') << "..." << std::string(lineWidth - 7, ' ') << "|" << std::endl;
    }

    // Close the table
    std::cout << headerBorder << std::endl;
}

void release_user_meta_at_batch_level(gpointer data, gpointer user_data)
{
    if (!data) return;

    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    if (!user_meta || !user_meta->user_meta_data) return;

    NvDsBbox3dObjectList *bbox3dList = (NvDsBbox3dObjectList *)user_meta->user_meta_data;
    if (!bbox3dList) return;

    // Validate num_objects to avoid out-of-bounds access
    if (bbox3dList->num_objects > MAX_NUM_OBJECTS) {
        g_printerr("Invalid num_objects: %u\n", bbox3dList->num_objects);
        return;
    }

    // Free sensorStr string if allocated
    if (bbox3dList->sensorStr) {
        g_free(bbox3dList->sensorStr);
        bbox3dList->sensorStr = nullptr;
    }

    // Loop through all the objects in the list
    for (guint objIdx = 0; objIdx < bbox3dList->num_objects; ++objIdx) {
        NvDsBbox3dObject &bboxObj = bbox3dList->objects[objIdx];

        // Free labels_3d string if allocated
        if (bboxObj.labels_3d) {
            g_free(bboxObj.labels_3d);
            bboxObj.labels_3d = nullptr;
        }

        // Free ocr_label string if allocated
        if (bboxObj.ocr_label) {
            g_free(bboxObj.ocr_label);
            bboxObj.ocr_label = nullptr;
        }

        // Ensure instance_feats is valid before accessing
        if (bboxObj.instance_feats) {
            if (bboxObj.instance_feats->embedding_vector) {
                g_free(bboxObj.instance_feats->embedding_vector);  // Free embedding vector with g_free
                bboxObj.instance_feats->embedding_vector = nullptr; // Set to null
            }
            g_free(bboxObj.instance_feats);  // Free instance_feats object with g_free
            bboxObj.instance_feats = nullptr; // Set to null
        }
    }

    // Free bbox3dList itself
    g_free(bbox3dList);
    user_meta->user_meta_data = nullptr; // Avoid dangling pointer
}