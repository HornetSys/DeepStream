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

#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvdsmeta_schema.h"

#include <torch/torch.h>

#include "instance_bank.h"
#include "decoder.h"
#include "bbox3d_utils.h"
#include "display_bbox3d.h"
#include "sparse_box_3d_generator.h"

#include "../utils/tensor.h"
#include "../utils/log_utils.h"
#include "../utils/tensor_init.h"
#include "../utils/config_properties.h"

/**
 * @brief Process output tensors for display and further analysis
 *
 * @param tensorCache TensorCache object
 * @param batch_meta DeepStream batch metadata for output annotation
 * @param config_properties Configuration properties used for processing
 * @param logger Logger for output messages
 * @return true if processing was successful, false otherwise
 */
bool postprocess_tensors(TensorCache* tensorCache,
                         NvDsBatchMeta* batch_meta,
                         const ConfigProperties& config_properties,
                         PrettyLogger* logger);

/**
 * @brief Copy user meta data from source to destination
 *
 * @param data Source user meta data
 * @param user_data User data
 * @return gpointer Destination user meta data
 */

gpointer copy_user_meta(gpointer data, gpointer user_data);

/**
 * @brief Release user meta data at batch level
 *
 * @param data User meta data
 * @param user_data User data
 */
void release_user_meta_at_batch_level(gpointer data, gpointer user_data);

/**
 * @brief Calculate mean timestamp from multiple frames in a batch
 *
 * @param batch_meta Batch metadata containing frame timestamps
 * @return guint64 Mean timestamp value
 */
guint64 calculate_mean_timestamp(NvDsBatchMeta* batch_meta);

/**
 * @brief Display the contents of a 3D bounding box object list to console
 *
 * This function displays detailed information about a NvDsBbox3dObjectList to the
 * standard output, including all object attributes and features.
 *
 * @param bbox3dList Pointer to the NvDsBbox3dObjectList to display
 * @param config_properties Configuration properties containing display settings
 */
void display_bbox3d_list(const NvDsBbox3dObjectList* bbox3dList, const ConfigProperties& config_properties);

#endif /* POSTPROCESSOR_H */