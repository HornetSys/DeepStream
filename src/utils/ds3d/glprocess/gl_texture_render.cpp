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

#include "ds3d/glprocess/gl_program.h"
#include "ds3d/glprocess/gl_buf.h"
#include "ds3d/common/hpp/yaml_config.hpp"

#include "gl_texture_render.h"
#include "gl_script.h"

namespace ds3d { namespace gl {

ErrCode
Texture3DRenderCtx::configure(const char* yamlConfig, uint32_t size)
{
    auto& tc = textrueConfig();
    std::string yamlStr = cppString(yamlConfig, size);
    DS3D_ERROR_RETURN(BaseRenderPriv::parseConfig(yamlStr, tc), "parse base config failed.");
    YAML::Node node = YAML::Load(yamlStr);
    auto ymaxVetexNum = node["max_vertex_num"];
    auto yTextureFrameKey = node["texture_frame_key"];
    auto yTextureVetexKey = node["texture_vetex_key"];
    auto yTextureCoordKey = node["texture_coord_key"];

    if (ymaxVetexNum) {
        uint32_t num = ymaxVetexNum.as<uint32_t>();
        if (num >= kTex2DVertDefaultNum) {
            tc.maxVertexNum = num;
        }
    }
    if (yTextureFrameKey) {
        tc.textureFrameKey = yTextureFrameKey.as<std::string>();
    }
    if (yTextureVetexKey) {
        tc.textureVertexKey = yTextureVetexKey.as<std::string>();
    }
    if (yTextureCoordKey) {
        tc.textureCoordKey = yTextureCoordKey.as<std::string>();
    }

    return ErrCode::kGood;
}

ErrCode
Texture3DRenderCtx::initResource()
{
    DS_ASSERT(getWindow());
    DS3D_ERROR_RETURN(
        buildShader(
            {std::string(GLSL_VERSION) + kImageTextureVertexScript,
             std::string(GLSL_VERSION) + kImageTextureFragmentScript},
            {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER}),
        "build texture 2D shader failed");

    const auto& tc = textrueConfig();
    initMvpFromConfig(tc);

    DS3D_ERROR_RETURN(storeViewPort(tc.viewLayout),
        "failed to store view port data");

    DS3D_ERROR_RETURN(initBase(tc), "failed to init base resource");
    DS3D_ERROR_RETURN(allocateBuffers(), "failed to allocate GL buffer data");
    return ErrCode::kGood;
}

ErrCode
Texture3DRenderCtx::uploadDefaultValues()
{
    static const vec3f kVertData[kTex2DVertDefaultNum] = {
        // buttorm right triangle
        {{-1.0f, -1.0f, 0.0f}},
        {{1.0f, -1.0f, 0.0f}},
        {{1.0f, 1.0f, 0.0f}},
        // top left triangle
        {{1.0f, 1.0f, 0.0f}},
        {{-1.0f, 1.0f, 0.0f}},
        {{-1.0f, -1.0f, 0.0f}},
    };
    static const vec2f kCoordData[kTex2DVertDefaultNum] = {
        // 1st uv coordinates for texture
        {{0.0f, 0.0f}},
        {{1.0f, 0.0f}},
        {{1.0f, 1.0f}},
        // 2nd uv coordinates for texture
        {{1.0f, 1.0f}},
        {{0.0f, 1.0f}},
        {{0.0f, 0.0f}},
    };

    DS_ASSERT(_vertexInUse == 0);

    DS3D_ERROR_RETURN(
        _vertexBuf->uploadSubData(0, sizeof(kVertData), &kVertData[0]),
        "upload texture3d vertex data failed");
    DS3D_ERROR_RETURN(
        _coordBuf->uploadSubData(0, sizeof(kCoordData), &kCoordData[0]),
        "upload texture3d vertex data failed");
    _vertexInUse = kTex2DVertDefaultNum;
    return ErrCode::kGood;
}

ErrCode
Texture3DRenderCtx::allocateBuffers()
{
    _vertexBuf.reset(VertexBuf::create(GL_ARRAY_BUFFER));
    DS3D_FAILED_RETURN(_vertexBuf, ErrCode::kGL, "Create texture3d vertexBuffer failed");
    DS3D_ERROR_RETURN(
        _vertexBuf->allocate<vec3f>(textrueConfig().maxVertexNum, GL_STREAM_DRAW),
        "allocate texture3d vertex buffer failed");

    _coordBuf.reset(VertexBuf::create(GL_ARRAY_BUFFER));
    DS3D_FAILED_RETURN(_coordBuf, ErrCode::kGL, "Create texture3d coordinate buffer failed");
    DS3D_ERROR_RETURN(
        _coordBuf->allocate<vec2f>(textrueConfig().maxVertexNum, GL_STREAM_DRAW),
        "allocate texture3d vertex uv buffer failed");

    DS_ASSERT(shader());
    DS3D_ERROR_RETURN(shader()->useProgram(), "use texture3d program failed");
    int vertAttrib = shader()->attribLocation("inVert");
    int uvAttrib = shader()->attribLocation("inTexCoord");

    DS3D_ERROR_RETURN(createVao(), "failed to create vao for texture3d render");
    DS_ASSERT(vao());

    vao()->bind();
    _vertexBuf->bind();
    glVertexAttribPointer(vertAttrib, 3, GL_FLOAT, GL_FALSE, sizeof(vec3f), (const void*)0);
    vao()->enableAttrib(vertAttrib);
    _vertexBuf->unbind();
    DS3D_RETURN_GL_ERROR("GL texture3d vao enable vertex attribute failed.");

    _coordBuf->bind();
    glVertexAttribPointer(uvAttrib, 2, GL_FLOAT, GL_FALSE, sizeof(vec2f), (const void*)0);
    vao()->enableAttrib(uvAttrib);
    _coordBuf->unbind();
    DS3D_RETURN_GL_ERROR("GL texture3d vao enable uv attribute failed.");

    vao()->unbind();
    shader()->closeProgram();
    return ErrCode::kGood;
}

ErrCode
Texture3DRenderCtx::allocateTexture(const Frame2DPlane& plane, MemType memType, FrameType frameType)
{
    DS_ASSERT(frameType == FrameType::kColorRGBA);
    DS3D_UNUSED(frameType);

    _texture.reset(Texture::create());
    DS3D_FAILED_RETURN(_texture, ErrCode::kGL, "Create image texture failed");
    auto preload = [this, &plane]() {
        glPixelStorei(GL_UNPACK_ROW_LENGTH, plane.pitchInBytes / plane.bytesPerPixel);
    };

    auto postload = [this]() {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    };

    DS3D_ERROR_RETURN(
        _texture->allocateImage(
            plane.width, plane.height, GL_UNSIGNED_BYTE, GL_RGBA8, GL_RGBA, preload, postload),
        "allocate color texture failed.");
    _texPlanes = plane;

    if (memType == MemType::kGpuCuda) {
        _cudaTexture.reset(new CudaTexture);
        DS3D_ERROR_RETURN(
            _cudaTexture->initCuda(_texture, 0), "Failed to register texture to cuda");
    }
    return ErrCode::kGood;
}

ErrCode
Texture3DRenderCtx::uploadImage(GuardDataMap& datamap)
{
    std::string keyName = textrueConfig().textureFrameKey;
    DS_ASSERT(datamap.hasData(keyName));
    Frame2DGuard imageFrame;
    DS3D_ERROR_RETURN(
        datamap.getGuardData(keyName, imageFrame),
        "No color image found in datamap during texture3d rendering");
    DS_ASSERT(imageFrame);
    MemType memType = imageFrame->memType();
    FrameType frameType = imageFrame->frameType();
    Frame2DPlane plane = imageFrame->getPlane(0);

    if (frameType != FrameType::kColorRGBA) {
        LOG_WARNING("Only RGBA image supported for texture3d rendering");
        return ErrCode::kUnsupported;
    }

    if (!_texture) {
        DS3D_ERROR_RETURN(
            allocateTexture(plane, memType, frameType),
            "allocate texture failed for texture3d render");
    }
    DS_ASSERT(_texture);
    DS_ASSERT(plane.bytesPerPixel == _texPlanes.bytesPerPixel);

    if (isCpuMem(memType)) {
        auto preload = [this, &plane]() {
            glPixelStorei(GL_UNPACK_ROW_LENGTH, plane.pitchInBytes / plane.bytesPerPixel);
        };

        auto postload = [this]() {
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        };

        DS3D_ERROR_RETURN(
            _texture->uploadImage(
                plane.width, plane.height, imageFrame->base(), GL_UNSIGNED_BYTE, GL_RGBA8, GL_RGBA,
                preload, postload),
            "upload depth color map failed.");
    } else if (memType == MemType::kGpuCuda) {
        DS3D_FAILED_RETURN(
            _cudaTexture, ErrCode::kGL,
            "gpu texture is not allocated, memtype mismatched between frames");
        DS_ASSERT(_cudaTexture);
        DS_ASSERT(_cudaStream);
        auto cudaProc = [&imageFrame, this](cudaArray_t dst, cudaStream_t stream) {
            uint32_t pixelBytes = (_texPlanes.bytesPerPixel ? _texPlanes.bytesPerPixel : 4);
            uint32_t pitch =
                (_texPlanes.pitchInBytes ? _texPlanes.pitchInBytes : pixelBytes * _texPlanes.width);
            DS3D_CHECK_CUDA_ERROR(
                cudaMemcpy2DToArrayAsync(
                    dst, 0, 0, imageFrame->base(), pitch, _texPlanes.width * pixelBytes,
                    _texPlanes.height, cudaMemcpyDeviceToDevice, stream),
                return ErrCode::kCuda, "cudaMemcpy2DToArray failed");
            return ErrCode::kGood;
        };
        _cudaTexture->cudaProcess(std::move(cudaProc), _cudaStream->get());
        // manually call syncData() to synchrnize cuda stream for performance
    }

    return ErrCode::kGood;
}

ErrCode
Texture3DRenderCtx::uploadVetex(GuardDataMap& datamap)
{
    const auto& tc = textrueConfig();
    if (!datamap.hasData(tc.textureVertexKey) || !datamap.hasData(tc.textureCoordKey)) {
        return ErrCode::kByPass;
    }
    FrameGuard vetexBuf;
    DS3D_ERROR_RETURN(
        datamap.getGuardData(tc.textureVertexKey, vetexBuf),
        "No vertex buffer is found in datamap during texture3d rendering");

    FrameGuard coordBuf;
    DS3D_ERROR_RETURN(
        datamap.getGuardData(tc.textureCoordKey, coordBuf),
        "No coordinate buffer is found in datamap during texture3d rendering");

    DS3D_FAILED_RETURN(
        vetexBuf->frameType() == FrameType::kPointXYZ && vetexBuf->dataType() == DataType::kFp32,
        ErrCode::kUnsupported, "texture3d render upload vetex only support kPointXYZ with FP32");
    DS3D_FAILED_RETURN(
        coordBuf->frameType() == FrameType::kPointCoordUV &&
            coordBuf->dataType() == DataType::kFp32,
        ErrCode::kUnsupported,
        "texture3d render upload coord only support kPointCoordUV with FP32");

    Shape vetShape = vetexBuf->shape();  // N x 3
    DS_ASSERT(vetShape.numDims == 2 && vetShape.d[1] == 3);  // PointXYZ
    int32_t numVet = (int32_t)vetShape.d[0];

    Shape cShape = coordBuf->shape();  // N x 2
    DS_ASSERT(cShape.numDims == 2 && cShape.d[1] == 2);  // PointColorCoord
    DS3D_FAILED_RETURN(
        numVet == cShape.d[0], ErrCode::kParam,
        "vertex buffer num: %d is different from uv coordinates num: %d", numVet, cShape.d[0]);
    numVet = std::min<int32_t>(numVet, tc.maxVertexNum);

    DS_ASSERT(_vertexBuf);

    DS3D_FAILED_RETURN(_vertexBuf, ErrCode::kGL, "Create vertexBuffer for points failed");
    DS3D_ERROR_RETURN(
        _vertexBuf->uploadSubData(0, sizeof(vec3f) * numVet, vetexBuf->base()),
        "upload texture3d vertex data failed");

    DS_ASSERT(_coordBuf);
    DS3D_ERROR_RETURN(
        _coordBuf->uploadSubData(0, sizeof(vec2f) * numVet, coordBuf->base()),
        "upload vertex data failed");

    _vertexInUse = numVet;
    return ErrCode::kGood;
}

ErrCode
Texture3DRenderCtx::uploadData(GuardDataMap datamap)
{
    const auto& tc = textrueConfig();
    // skip datamap without image frames
    if (!datamap.hasData(tc.textureFrameKey)) {
        return ErrCode::kByPass;
    }
    DS3D_ERROR_RETURN(uploadImage(datamap), "Failed to upload image into texture3d.");

    if (datamap.hasData(tc.textureVertexKey) && datamap.hasData(tc.textureCoordKey)) {
        DS3D_ERROR_RETURN(uploadVetex(datamap), "Failed to upload vetex data into texture3d.");
    }

    if (!_vertexInUse) {
        DS3D_ERROR_RETURN(uploadDefaultValues(), "Failed to upload default vertex values.");
    }
    return ErrCode::kGood;
}

ErrCode
Texture3DRenderCtx::draw(GuardDataMap datamap)
{
    DS3D_UNUSED(datamap);
    int32_t x = 0, y = 0;
    uint32_t w = 0, h = 0;
    DS3D_ERROR_RETURN(calculateViewPort(x, y, w, h), "Failed to calculate view port values");

    glViewport(x, y, w, h);
    glScissor(x, y, w, h);
    glEnable(GL_SCISSOR_TEST);
    if (textrueConfig().is2dOnly) {
        glDisable(GL_DEPTH_TEST);
    } else {
        glEnable(GL_DEPTH_TEST);
    }
    if (textrueConfig().needColorClear) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    Ptr<Program> p = shader();
    DS_ASSERT(p);

    DS3D_ERROR_RETURN(p->useProgram(), "use texture3d program failed");
    mat4f mvp = mvpMat(model(), view(), project());
    glUniformMatrix4fv(p->uniformLocation("mvp"), 1, GL_FALSE, (const GLfloat*)&mvp[0]);
    glUniform1i(p->uniformLocation("imageTexture"), 0);
    DS_ASSERT(_texture);
    glActiveTexture(GL_TEXTURE0);
    _texture->bind(GL_TEXTURE_2D);
    DS3D_RETURN_GL_ERROR("binding texture for render failed.");

    DS_ASSERT(vao());
    vao()->bind();
    glDrawArrays(GL_TRIANGLES, 0, _vertexInUse);
    DS3D_RETURN_GL_ERROR("draw image texture failed.");
    vao()->unbind();
    _texture->unbind(GL_TEXTURE_2D);
    glDisable(GL_SCISSOR_TEST);
    p->closeProgram();

    return ErrCode::kGood;
}

ErrCode
Texture3DRenderCtx::flush()
{
    _vertexInUse = 0;
    return ErrCode::kGood;
}

ErrCode
Texture3DRenderCtx::deinitResource()
{
    syncData();
    _cudaTexture.reset();
    _texture.reset();
    _coordBuf.reset();
    _vertexBuf.reset();
    _vertexInUse = 0;
    return ErrCode::kGood;
}

}}  // namespace ds3d::gl

using namespace ds3d;

DS3D_EXTERN_C_BEGIN

gl::iGLRender*
NvDs3D_CreateTexture3DRender(const char* yamlConfig, uint32_t size)
{
    auto render = std::make_unique<gl::Texture3DRenderCtx>();
    DS3D_FAILED_RETURN(render, nullptr, "create texture 3D render failed.");
    return render.release();
}

DS3D_EXTERN_C_END