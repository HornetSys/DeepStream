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
 * @file triton_client.cpp
 * @brief Triton Inference Server C++ client wrapper for C API
 */

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "grpc_client.h"

namespace tc = triton::client;

/**
 * Triton client wrapper class
 */
class TritonClientWrapper {
public:
    TritonClientWrapper(const std::string& url, const std::string& model_name)
        : url_(url), model_name_(model_name), input_width_(0), input_height_(0),
          embedding_dim_(0) {}

    ~TritonClientWrapper() {
        client_.reset();
    }

    bool Initialize() {
        // Create GRPC client
        tc::Error err = tc::InferenceServerGrpcClient::Create(&client_, url_, false);
        if (!err.IsOk()) {
            std::cerr << "Error creating Triton client: " << err << std::endl;
            return false;
        }

        // Check server health
        bool live = false;
        err = client_->IsServerLive(&live);
        if (!err.IsOk() || !live) {
            std::cerr << "Triton server not available at " << url_ << std::endl;
            return false;
        }

        // Get model metadata
        std::string model_metadata;
        err = client_->ModelMetadata(&model_metadata, model_name_);
        if (!err.IsOk()) {
            std::cerr << "Failed to get model metadata: " << err << std::endl;
            return false;
        }

        // Get model config
        std::string model_config;
        err = client_->ModelConfig(&model_config, model_name_);
        if (!err.IsOk()) {
            std::cerr << "Failed to get model config: " << err << std::endl;
            return false;
        }

        // Parse dimensions (simplified - assumes specific model structure)
        input_width_ = 224;   // Default for RADIO-CLIP
        input_height_ = 224;
        embedding_dim_ = 1024;

        std::cout << "Triton client initialized: " << url_ << ", model: " << model_name_ << std::endl;
        std::cout << "Input: " << input_width_ << "x" << input_height_ 
                  << ", Embedding dim: " << embedding_dim_ << std::endl;

        return true;
    }

    bool Infer(float* input_data, int num_images, float* output_embeddings) {
        if (!client_) {
            std::cerr << "Client not initialized" << std::endl;
            return false;
        }

        // Prepare input
        std::vector<int64_t> input_shape = {num_images, 3, input_height_, input_width_};
        tc::InferInput* input;
        tc::Error err = tc::InferInput::Create(&input, "pixel_values", input_shape, "FP32");
        if (!err.IsOk()) {
            std::cerr << "Failed to create input: " << err << std::endl;
            return false;
        }

        // Set input data
        size_t input_byte_size = num_images * 3 * input_height_ * input_width_ * sizeof(float);
        err = input->AppendRaw(reinterpret_cast<uint8_t*>(input_data), input_byte_size);
        if (!err.IsOk()) {
            std::cerr << "Failed to set input data: " << err << std::endl;
            delete input;
            return false;
        }

        // Prepare output
        tc::InferRequestedOutput* output;
        err = tc::InferRequestedOutput::Create(&output, "image_embeddings");
        if (!err.IsOk()) {
            std::cerr << "Failed to create output: " << err << std::endl;
            delete input;
            return false;
        }

        // Prepare options
        tc::InferOptions options(model_name_);
        options.model_version_ = "";

        // Run inference
        std::vector<tc::InferInput*> inputs = {input};
        std::vector<const tc::InferRequestedOutput*> outputs = {output};
        tc::InferResult* result;

        err = client_->Infer(&result, options, inputs, outputs);
        if (!err.IsOk()) {
            std::cerr << "Inference failed: " << err << std::endl;
            delete input;
            delete output;
            return false;
        }

        // Get output data
        const uint8_t* output_data;
        size_t output_byte_size;
        err = result->RawData("image_embeddings", &output_data, &output_byte_size);
        if (!err.IsOk()) {
            std::cerr << "Failed to get output data: " << err << std::endl;
            delete result;
            delete input;
            delete output;
            return false;
        }

        // Copy to output buffer
        memcpy(output_embeddings, output_data, output_byte_size);

        // Cleanup
        delete result;
        delete input;
        delete output;

        return true;
    }

    unsigned int GetInputWidth() const { return input_width_; }
    unsigned int GetInputHeight() const { return input_height_; }
    unsigned int GetEmbeddingDim() const { return embedding_dim_; }

private:
    std::string url_;
    std::string model_name_;
    std::unique_ptr<tc::InferenceServerGrpcClient> client_;
    unsigned int input_width_;
    unsigned int input_height_;
    unsigned int embedding_dim_;
};

// C API implementations

extern "C" {

void* nvds_triton_client_create(const char* url, const char* model_name) {
    TritonClientWrapper* client = new TritonClientWrapper(url, model_name);
    if (!client->Initialize()) {
        delete client;
        return nullptr;
    }
    return static_cast<void*>(client);
}

void nvds_triton_client_destroy(void* client) {
    if (client) {
        delete static_cast<TritonClientWrapper*>(client);
    }
}

int nvds_triton_client_get_input_size(void* client, unsigned int* width, unsigned int* height) {
    if (!client || !width || !height) {
        return -1;
    }
    TritonClientWrapper* wrapper = static_cast<TritonClientWrapper*>(client);
    *width = wrapper->GetInputWidth();
    *height = wrapper->GetInputHeight();
    return 0;
}

int nvds_triton_client_get_embedding_dim(void* client) {
    if (!client) {
        return 0;
    }
    TritonClientWrapper* wrapper = static_cast<TritonClientWrapper*>(client);
    return wrapper->GetEmbeddingDim();
}

int nvds_triton_client_infer(void* client, float* input_data, int num_images,
                              float* output_embeddings) {
    if (!client || !input_data || !output_embeddings || num_images <= 0) {
        return -1;
    }
    TritonClientWrapper* wrapper = static_cast<TritonClientWrapper*>(client);
    return wrapper->Infer(input_data, num_images, output_embeddings) ? 0 : -1;
}

} // extern "C"
