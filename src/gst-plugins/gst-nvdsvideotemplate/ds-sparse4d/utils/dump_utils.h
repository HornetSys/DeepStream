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

#ifndef DUMP_UTILS_H
#define DUMP_UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "log_utils.h"
#include <sys/stat.h>
#include <unistd.h>
#include "tensor.h"

/**
 * @brief Dumps NvBufSurface data to file
 *
 * This function dumps the raw frame data from an NvBufSurface to a file.
 * The file format is determined by the video format of the surface.
 *
 * @param in_surface The NvBufSurface to dump
 * @param batch_meta Metadata associated with the surface
 * @param image_name Base name for output files
 * @param frame_num Current frame number for naming
 * @param video_fmt Input video format enumeration
 * @param scratch_nvbuf_surface Scratch buffer for copying device memory to host
 * @param batched_mode True to create a single tiled image with all batches, false for individual files
 * @param logger PrettyLogger instance for logging dump operations
 */
void DumpNvBufSurface(NvBufSurface *in_surface,
                      NvDsBatchMeta *batch_meta,
                      const std::string& image_name,
                      guint frame_num,
                      GstVideoFormat video_fmt,
                      void*& scratch_nvbuf_surface,
                      bool batched_mode = true,
                      PrettyLogger* logger = nullptr);

/**
 * @brief Dump tensor as a raw data file
 *
 * This function dumps the tensor data to a raw file format with appropriate extensions
 * based on the tensor data type and channel count.
 *
 * @param tensor The Tensor structure containing all tensor information
 * @param image_name Base name for the output file
 * @param batched_mode True to create a single tiled image with all batches, false for individual files
 * @param logger PrettyLogger instance for logging dump operations
 * @return bool True if successfully dumped, false otherwise
 */
bool dump_tensor(const Tensor& tensor,
                const std::string& image_name,
                bool batched_mode = true,
                PrettyLogger* logger = nullptr);

#endif /* DUMP_UTILS_H */