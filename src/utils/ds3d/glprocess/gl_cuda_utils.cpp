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

#include "ds3d/common/common.h"
#include "ds3d/glprocess/gl_common.h"

#include "ds3d/glprocess/gl_cuda_utils.h"

namespace ds3d { namespace gl {

CudaTexture*
CudaTexture::create(int width, int height, GLenum type, int internalFormat, int format)
{
    Ptr<Texture> texture(Texture::create());
    DS3D_FAILED_RETURN(texture, nullptr, "create texture for cuda failed");
    DS3D_FAILED_RETURN(
        isGood(texture->allocateImage(width, height, type, internalFormat, format)), nullptr,
        "allocate texture (w: %d, h: %d) buffer for cuda failed", width, height);

    auto cuTex = std::make_unique<CudaTexture>();
    DS3D_FAILED_RETURN(
        isGood(cuTex->initCuda(std::move(texture))), nullptr,
        "allocate texture (w: %d, h: %d) buffer for cuda failed", width, height);
    return cuTex.release();
}

/* Flag selected from [cudaGraphicsRegisterFlagsNone, cudaGraphicsRegisterFlagsWriteDiscard]
 */
ErrCode
CudaTexture::initCuda(Ptr<Texture> t, uint32_t flag)
{
    DS_ASSERT(t && t->get());
    cudaGraphicsResource* cudaRes = nullptr;
    DS3D_CHECK_CUDA_ERROR(
        cudaGraphicsGLRegisterImage(&cudaRes, t->get(), GL_TEXTURE_2D, flag), return ErrCode::kCuda,
        "register gl texture into cuda failed");

    _texture = std::move(t);
    _cudaRes = cudaRes;

    return ErrCode::kGood;
}

ErrCode
CudaTexture::cudaProcess(
    std::function<ErrCode(cudaArray_t, cudaStream_t)> cudaOps, cudaStream_t stream)
{
    DS_ASSERT(_texture && _texture->get());
    DS_ASSERT(_cudaRes);

    cudaArray* cudaArray = nullptr;
    DS3D_CHECK_CUDA_ERROR(
        cudaGraphicsMapResources(1, &cudaRes(), stream), return ErrCode::kCuda,
        "cudaGraphicsMapResources failed on texture");
    DS3D_CHECK_CUDA_ERROR(
        cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaRes(), 0, 0), return ErrCode::kCuda,
        "cudaGraphicsSubResourceGetMappedArray failed on texture");

    DS_ASSERT(cudaArray);
    DS_ASSERT(cudaOps);
    ErrCode c = cudaOps(cudaArray, stream);
    DS3D_ERROR_RETURN(c, "cuda operation failed on cuda array");

    DS3D_CHECK_CUDA_ERROR(
        cudaGraphicsUnmapResources(1, &cudaRes(), stream), return ErrCode::kCuda,
        "cudaGraphicsUnmapResources failed on texture");

    return ErrCode::kGood;
}

CudaVertexBuf::~CudaVertexBuf()
{
    if (_cudaRes) {
        cudaGraphicsUnregisterResource(_cudaRes);
        _cudaRes = nullptr;
    }
    _vertexBuf.reset();
}

ErrCode
CudaVertexBuf::initCuda(Ptr<VertexBuf> t, uint32_t flag)
{
    DS_ASSERT(t && t->get());
    cudaGraphicsResource* cudaRes = nullptr;
    DS3D_CHECK_CUDA_ERROR(
        cudaGraphicsGLRegisterBuffer(&cudaRes, t->get(), flag), return ErrCode::kCuda,
        "register gl vertex buf into cuda failed");

    _vertexBuf = std::move(t);
    _cudaRes = cudaRes;

    return ErrCode::kGood;
}


ErrCode
CudaVertexBuf::cudaProcess(
    std::function<ErrCode(void*, size_t, cudaStream_t)> cudaOps, cudaStream_t stream)
{
    DS_ASSERT(_vertexBuf && _vertexBuf->get());
    DS_ASSERT(_cudaRes);

    void* cudaPtr = nullptr;
    size_t numBytes = 0;
    DS3D_CHECK_CUDA_ERROR(
        cudaGraphicsMapResources(1, &cudaRes(), stream), return ErrCode::kCuda,
        "cudaGraphicsMapResources failed on texture");
    DS3D_CHECK_CUDA_ERROR(
        cudaGraphicsResourceGetMappedPointer(&cudaPtr, &numBytes, cudaRes()), return ErrCode::kCuda,
        "cudaGraphicsSubResourceGetMappedArray failed on texture");

    DS_ASSERT(cudaPtr);
    DS_ASSERT(cudaOps);
    ErrCode c = cudaOps(cudaPtr, numBytes, stream);
    DS3D_ERROR_RETURN(c, "cuda operation failed on cuda array");

    DS3D_CHECK_CUDA_ERROR(
        cudaGraphicsUnmapResources(1, &cudaRes(), stream), return ErrCode::kCuda,
        "cudaGraphicsUnmapResources failed on texture");

    return ErrCode::kGood;
}

}}  // namespace ds3d::gl