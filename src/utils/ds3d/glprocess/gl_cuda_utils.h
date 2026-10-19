/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DS3D_GL_PROCESS_GL_CUDA_UTILSS_H
#define DS3D_GL_PROCESS_GL_CUDA_UTILSS_H

#include "ds3d/common/common.h"
#include "ds3d/glprocess/gl_common.h"
#include "ds3d/glprocess/gl_buf.h"
#include "ds3d/common/helper/cuda_utils.h"

#include <cuda.h>
#include <cuda_gl_interop.h>

namespace ds3d { namespace gl {

class CudaTexture {
    Ptr<Texture> _texture;
    cudaGraphicsResource* _cudaRes = nullptr;

public:
    static CudaTexture* create(int width, int height, GLenum type, int internalFormat, int format);

    CudaTexture() = default;
    ~CudaTexture() = default;
    ErrCode initCuda(Ptr<Texture> t, uint32_t flag = 0);

    const Ptr<Texture>& texture() const { return _texture; }
    cudaGraphicsResource* cudaRes() const { return _cudaRes; }
    cudaGraphicsResource*& cudaRes() { return _cudaRes; }
    ErrCode cudaProcess(
        std::function<ErrCode(cudaArray_t, cudaStream_t)> cudaOps, cudaStream_t stream = 0);
};

class CudaVertexBuf {
    Ptr<VertexBuf> _vertexBuf;
    cudaGraphicsResource* _cudaRes = nullptr;

public:
    CudaVertexBuf() = default;
    ~CudaVertexBuf();
    ErrCode initCuda(Ptr<VertexBuf> t, uint32_t flag = 0);

    const Ptr<VertexBuf>& glBuf() const { return _vertexBuf; }
    cudaGraphicsResource* cudaRes() const { return _cudaRes; }
    cudaGraphicsResource*& cudaRes() { return _cudaRes; }
    ErrCode cudaProcess(
        std::function<ErrCode(void*, size_t, cudaStream_t)> cudaOps, cudaStream_t stream = 0);
};

}}  // namespace ds3d::gl

#endif  // DS3D_GL_PROCESS_GL_CUDA_UTILSS_H