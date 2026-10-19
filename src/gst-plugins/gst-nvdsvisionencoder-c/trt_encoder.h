/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __TRT_ENCODER_H__
#define __TRT_ENCODER_H__

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create TensorRT encoder client
 * @param url Engine path (if contains .plan/.engine) or Triton URL (fallback)
 * @param model_name Model name for path construction
 * @param batch_size Batch size for engine building
 * @param onnx_model Path to ONNX model for building engine if not found
 * @return Opaque handle to encoder, or NULL on failure
 */
void* nvds_triton_client_create(const char* url, const char* model_name,
                                 int batch_size, const char* onnx_model);

/**
 * Destroy TensorRT encoder client
 * @param client Opaque handle from nvds_triton_client_create
 */
void nvds_triton_client_destroy(void* client);

/**
 * Get input dimensions from encoder
 * @param client Opaque handle
 * @param width Output: input width
 * @param height Output: input height
 * @return 0 on success, -1 on failure
 */
int nvds_triton_client_get_input_size(void* client, unsigned int* width, unsigned int* height);

/**
 * Get embedding dimension from encoder
 * @param client Opaque handle
 * @return Embedding dimension, or 0 on failure
 */
int nvds_triton_client_get_embedding_dim(void* client);

/**
 * Get maximum batch size the loaded engine supports (for runtime model validation)
 * @param client Opaque handle from nvds_triton_client_create
 * @return max batch size, or 0 if it cannot be determined
 */
unsigned int nvds_triton_client_get_max_batch_size(void* client);

/**
 * Get input width
 * @param client Opaque handle
 * @return Input width, or 0 on failure
 */
unsigned int nvds_triton_client_get_input_width(void* client);

/**
 * Get input height
 * @param client Opaque handle
 * @return Input height, or 0 on failure
 */
unsigned int nvds_triton_client_get_input_height(void* client);

/**
 * Run inference on host input data
 * @param client Opaque handle
 * @param input_data Host pointer to input tensor [N, C, H, W]
 * @param num_images Number of images in batch
 * @param output_embeddings Host pointer for output [N, embedding_dim]
 * @return 0 on success, -1 on failure
 */
int nvds_triton_client_infer(void* client, float* input_data, int num_images,
                              float* output_embeddings);

/**
 * Run inference on device input data (GPU path)
 * @param client Opaque handle
 * @param d_input_data Device pointer to input tensor [N, C, H, W]
 * @param num_images Number of images in batch
 * @param output_embeddings Host pointer for output [N, embedding_dim]
 * @return 0 on success, -1 on failure
 */
int nvds_triton_client_infer_device(void* client, float* d_input_data, int num_images,
                                     float* output_embeddings);

#ifdef __cplusplus
}
#endif

#endif /* __TRT_ENCODER_H__ */
