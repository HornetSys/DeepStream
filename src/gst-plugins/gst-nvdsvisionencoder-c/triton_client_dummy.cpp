/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file triton_client_dummy.cpp
 * @brief PHASE 1: Dummy Triton client for testing C plugin metadata attachment
 * 
 * This generates deterministic dummy embeddings to verify the C plugin works correctly.
 * Once verified, this will be replaced with real Triton inference.
 */

#include <cstdio>
#include <cstring>
#include <string>

/**
 * Dummy client structure for Phase 1 testing
 */
struct DummyTritonClient {
    std::string url;
    std::string model_name;
    unsigned int input_width;
    unsigned int input_height;
    unsigned int embedding_dim;
    int inference_count;
};

extern "C" {

/**
 * Create a dummy Triton client for testing
 */
void* nvds_triton_client_create(const char* url, const char* model_name) {
    DummyTritonClient* client = new DummyTritonClient();
    client->url = url ? url : "localhost:8001";
    client->model_name = model_name ? model_name : "radio-clip";
    client->input_width = 384;    // RADIO-CLIP input size
    client->input_height = 384;
    client->embedding_dim = 1152; // RADIO-CLIP embedding dimension
    client->inference_count = 0;
    
    printf("[TritonClientDummy] ==========================================\n");
    printf("[TritonClientDummy] PHASE 1: Testing with DUMMY embeddings\n");
    printf("[TritonClientDummy] URL: %s, Model: %s\n", client->url.c_str(), client->model_name.c_str());
    printf("[TritonClientDummy] Input: %ux%u, Embedding dim: %u\n",
           client->input_width, client->input_height, client->embedding_dim);
    printf("[TritonClientDummy] ==========================================\n");
    
    return client;
}

/**
 * Destroy dummy client
 */
void nvds_triton_client_destroy(void* client) {
    if (client) {
        DummyTritonClient* tc = static_cast<DummyTritonClient*>(client);
        printf("[TritonClientDummy] Destroying client. Total inferences: %d\n", tc->inference_count);
        delete tc;
    }
}

/**
 * Get model input dimensions
 */
int nvds_triton_client_get_input_size(void* client, unsigned int* width, unsigned int* height) {
    if (!client || !width || !height) {
        return -1;
    }
    
    DummyTritonClient* tc = static_cast<DummyTritonClient*>(client);
    *width = tc->input_width;
    *height = tc->input_height;
    return 0;
}

/**
 * Get embedding dimension
 */
int nvds_triton_client_get_embedding_dim(void* client) {
    if (!client) {
        return 0;
    }
    
    DummyTritonClient* tc = static_cast<DummyTritonClient*>(client);
    return tc->embedding_dim;
}

/**
 * Perform "inference" - generate dummy embeddings for testing
 * 
 * Each object gets a unique, deterministic pattern that we can verify in JSON output:
 * - First 10 values: object index (so we can identify which object)
 * - Remaining values: sequential pattern (for verification)
 */
int nvds_triton_client_infer(void* client, float* input_data, int num_images,
                              float* output_embeddings) {
    if (!client || !output_embeddings || num_images <= 0) {
        return -1;
    }
    
    DummyTritonClient* tc = static_cast<DummyTritonClient*>(client);
    tc->inference_count++;
    
    // Generate deterministic embeddings for each image
    for (int i = 0; i < num_images; i++) {
        float* emb = output_embeddings + (i * tc->embedding_dim);
        
        // Unique pattern: first 10 values are object index, rest are sequential
        for (unsigned int j = 0; j < tc->embedding_dim; j++) {
            if (j < 10) {
                emb[j] = (float)(i + 100);  // Object marker (100, 101, 102, ...)
            } else if (j < 20) {
                emb[j] = (float)(tc->inference_count);  // Batch marker
            } else {
                emb[j] = (float)j * 0.001f;  // Sequential pattern
            }
        }
    }
    
    if (tc->inference_count <= 3) {
        printf("[TritonClientDummy] Inference #%d: Generated dummy embeddings for %d images (dim=%u)\n",
               tc->inference_count, num_images, tc->embedding_dim);
        printf("[TritonClientDummy]   Pattern: first 10 values = object_id+100, next 10 = batch_num\n");
    }
    
    return 0;  // Success
}

} // extern "C"
