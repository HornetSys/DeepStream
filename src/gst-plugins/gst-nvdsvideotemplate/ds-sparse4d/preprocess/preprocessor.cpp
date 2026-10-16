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

#include <string>
#include <vector>
#include <cuda_runtime_api.h>

#include "gstnvdspreprocess_allocator.h"
#include "nvdspreprocess_interface.h"
#include "nvdspreprocess_meta.h"
#include "nvdsmeta.h"
#include "gstnvdspreprocess.h"

#include "preprocessor.h"

Tensor* parse_preprocess_batch_meta(NvDsBatchMeta *batch_meta, PrettyLogger* logger) {

    Tensor* preprocessed_tensor = new Tensor();
    // Iterate through user metadata in batch metadata
    for (NvDsMetaList *l_user = batch_meta->batch_user_meta_list; l_user != NULL; l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user->data);
        // Check if this is preprocess batch metadata
        if (user_meta->base_meta.meta_type == NVDS_PREPROCESS_BATCH_META) {
            logger->info("Found NVDS_PREPROCESS_BATCH_META");
            // Cast the user meta data to the preprocess batch meta type
            GstNvDsPreProcessBatchMeta *preproc_meta = (GstNvDsPreProcessBatchMeta *)user_meta->user_meta_data;

            // Set basic tensor properties
            preprocessed_tensor->buffer = preproc_meta->tensor_meta->raw_tensor_buffer;
            preprocessed_tensor->size = preproc_meta->tensor_meta->buffer_size;
            preprocessed_tensor->data_type = preproc_meta->tensor_meta->data_type;
            preprocessed_tensor->gpu_id = preproc_meta->tensor_meta->gpu_id;

            // Create a new tensor shape with an extra dimension of 1 at the front (BNCHW)
            std::vector<int> new_tensor_shape;
            new_tensor_shape.push_back(1);  // Add dimension B=1 at the front

            // Copy the original NCHW dimensions
            for (const auto& dim : preproc_meta->tensor_meta->tensor_shape) {
                new_tensor_shape.push_back(dim);
            }

            // Set the new tensor shape
            preprocessed_tensor->shape = new_tensor_shape;

            // Extract dimensions from tensor shape (now BNCHW)
            if (new_tensor_shape.size() >= 5) {
                logger->info("Tensor Shape: " +
                                std::to_string(preprocessed_tensor->shape[0]) + "x" +
                                std::to_string(preprocessed_tensor->shape[1]) + "x" +
                                std::to_string(preprocessed_tensor->shape[2]) + "x" +
                                std::to_string(preprocessed_tensor->shape[3]) + "x" +
                                std::to_string(preprocessed_tensor->shape[4]));

            } else {
                logger->warning("Invalid tensor shape dimensions in NVDS_PREPROCESS_BATCH_META");
            }

            // Set tensor name if available
            if (!preproc_meta->tensor_meta->tensor_name.empty()) {
                preprocessed_tensor->name = preproc_meta->tensor_meta->tensor_name;
                logger->info("Tensor Name: " + preprocessed_tensor->name);
            }

            // Set the frame number if ROI vector is available
            if (!preproc_meta->roi_vector.empty() && preproc_meta->roi_vector[0].frame_meta) {
                preprocessed_tensor->frame_num = preproc_meta->roi_vector[0].frame_meta->frame_num;
            }

            return preprocessed_tensor;
        }
    }

    delete preprocessed_tensor;
    return nullptr;
}