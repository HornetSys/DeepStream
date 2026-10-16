/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ds3d/common/hpp/frame.hpp"

#include "gl_depth_render_ctx.h"
#include "glfw_init.h"

namespace ds3d { namespace gl {

namespace {
static constexpr size_t kVerTexNum = 6;

const std::string kVertexTexScript = {
    "#version 300 es\n"
    "precision mediump float;\n"
    "layout (location = 0) in vec3 inVert;\n"
    "layout (location = 2) in vec2 inTexCoord;\n"
    "\n"
    "out vec2 TexCoord;\n"
    "\n"
    "void main()\n"
    "{\n"
    "	gl_Position = vec4(inVert, 1.0);\n"
    "	TexCoord = vec2(inTexCoord.x, inTexCoord.y);\n"
    "}\n"};

const std::string kImageFragmentScript = {
    "#version 300 es\n"
    "precision mediump float;\n"
    "in vec2 TexCoord;\n"
    "out vec4 FragColor;\n"
    "uniform sampler2D imageTexture;\n"
    "\n"
    "void main()\n"
    "{\n"
    "	vec2 finalCoord = vec2(TexCoord.x, 1.0 - TexCoord.y);\n"
    "	FragColor = texture(imageTexture, finalCoord);\n"
    "}\n"};

// depthTexture
// glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, width, height, 0, GL_RG, GL_UNSIGNED_BYTE, data);
// colorSampler, width = size, height = 1
// glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, size, 1, 0, GL_RGB,  GL_UNSIGNED_BYTE, data);
// depthStep = 1.0f/(maxDepth - minDepth);
const std::string kDepthFragmentScript = {
    "#version 300 es\n"
    "precision mediump float;\n"
    "in vec2 TexCoord;\n"
    "out vec4 FragColor;\n"
    "uniform sampler2D depthTexture; // GL_RG \n"
    "uniform sampler2D colorSampler; // GL_RGB \n"
    "uniform float depthScale;\n"
    "uniform float minDepth;\n"
    "uniform float depthStep;\n"
    "\n"
    "void main() {\n"
    "    vec2 depthCoord = vec2(TexCoord.x, 1.0 - TexCoord.y);\n"
    "    vec4 dVal = texture(depthTexture, depthCoord);\n"
    "    float d = (dVal.x + dVal.y * 256.0) * 256.0;\n"
    "    d = clamp(d, 0.0f, 65535.0f) * depthScale;\n"
    "    float colorX = (d - minDepth) * depthStep;\n"
    "    colorX = clamp(colorX, 0.0f, 0.9999f);\n"
    "    FragColor = texture(colorSampler, vec2(colorX, 0.0));\n"
    "}\n"};

}  // namespace

ErrCode
DepthRenderCtxImpl::initResource()
{
    if (!_window) {
        _window.reset(NvDs3D_CreateGlfwWindow(1280, 720, "video-render"));
        DS3D_FAILED_RETURN(
            _window, ErrCode::kGL, "create glfw window failed for resource creation");
    }

    DS_ASSERT(_window);
    _window->getSize(_initWindowInfo.width, _initWindowInfo.height);
    _initWindowInfo.title = _window->getTitle();
    if (_imageProgInfo.enabled) {
        std::unique_ptr<Program> imageProg(BuildProgramFromSource(
            {kVertexTexScript, kImageFragmentScript}, {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER}));
        DS3D_FAILED_RETURN(imageProg, ErrCode::kGL, "Build Image shaders failed");
        _imageProgInfo.program = std::move(imageProg);
        DS_ASSERT(imageProgram());
    }

    if (_depthProgInfo.enabled) {
        std::unique_ptr<Program> depthProg(BuildProgramFromSource(
            {kVertexTexScript, kDepthFragmentScript}, {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER}));
        DS3D_FAILED_RETURN(depthProg, ErrCode::kGL, "Build Depth shaders failed");
        _depthProgInfo.program = std::move(depthProg);
        DS_ASSERT(depthProgram());
    }

    DS3D_ERROR_RETURN(initTextures(), "init Textures failed.");

    DS3D_ERROR_RETURN(initVertexBuffer(), "init Vertex Buffers failed");

    DS3D_ERROR_RETURN(initVAO(), "init all VAO failed");

    return ErrCode::kGood;
}

ErrCode
DepthRenderCtxImpl::initVertexBuffer()
{
    vec3f vertData[kVerTexNum] = {
        // first triangle
        {{-1.0f, -1.0f, 0.0f}},
        {{1.0f, -1.0f, 0.0f}},
        {{1.0f, 1.0f, 0.0f}},
        // second triangle
        {{1.0f, 1.0f, 0.0f}},
        {{-1.0f, 1.0f, 0.0f}},
        {{-1.0f, -1.0f, 0.0f}},
    };
    vec2f coordData[kVerTexNum] = {
        // first triangle
        {{0.0f, 0.0f}},
        {{1.0f, 0.0f}},
        {{1.0f, 1.0f}},
        // second triangle
        {{1.0f, 1.0f}},
        {{0.0f, 1.0f}},
        {{0.0f, 0.0f}},
    };

    _vertexBuf.reset(VertexBuf::create(GL_ARRAY_BUFFER));
    DS3D_FAILED_RETURN(_vertexBuf, ErrCode::kGL, "Create vertexBuffer failed");
    DS3D_ERROR_RETURN(
        _vertexBuf->upload<vec3f>(kVerTexNum, &vertData[0], GL_STATIC_DRAW),
        "upload vertex data failed");

    _coordBuf.reset(VertexBuf::create(GL_ARRAY_BUFFER));
    DS3D_FAILED_RETURN(_coordBuf, ErrCode::kGL, "Create coordinate buffer failed");
    DS3D_ERROR_RETURN(
        _coordBuf->upload<vec2f>(kVerTexNum, &coordData[0], GL_STATIC_DRAW),
        "upload vertex data failed");

    return ErrCode::kGood;
}

ErrCode
DepthRenderCtxImpl::initVAO()
{
    auto initf = [this](ProgramInfo& vao) {
        if (!vao.enabled) {
            return ErrCode::kGood;
        }
        vao.vao.reset(VertexArray::create());
        DS3D_FAILED_RETURN(vao.vao, ErrCode::kGL, "Create VAO failed");

        Program* p = vao.program.get();
        DS_ASSERT(p);
        DS3D_ERROR_RETURN(p->useProgram(), "use program failed");
        vao.inTexCoordAttrib = p->attribLocation("inTexCoord");
        vao.inVertAttrib = p->attribLocation("inVert");

        vao.vao->bind();
        _vertexBuf->bind();
        vao.vao->enableAttrib(vao.inVertAttrib);
        // stride: 0 for tightly packed
        glVertexAttribPointer(vao.inVertAttrib, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        _vertexBuf->unbind();
        DS3D_RETURN_GL_ERROR("enable inVert into vao failed.");

        _coordBuf->bind();
        vao.vao->enableAttrib(vao.inTexCoordAttrib);
        // stride: 0 for tightly packed
        glVertexAttribPointer(vao.inTexCoordAttrib, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        _coordBuf->unbind();
        vao.vao->unbind();
        DS3D_RETURN_GL_ERROR("enable inTexCoord into vao failed.");
        return ErrCode::kGood;
    };

    DS3D_ERROR_RETURN(initf(_imageProgInfo), "init image vao failed");

    DS3D_ERROR_RETURN(initf(_depthProgInfo), "init depth vao failed");
    return ErrCode::kGood;
}

ErrCode
DepthRenderCtxImpl::initTextures()
{
    std::vector<vec4b> colorMap;
    vec4b start = {{_minDepthColor.x, _minDepthColor.y, _minDepthColor.z, 255}};
    vec4b end = {{_maxDepthColor.x, _maxDepthColor.y, _maxDepthColor.z, 255}};
    colorMap.resize(64);
    for (size_t i = 0; i < colorMap.size(); ++i) {
        float a = (colorMap.size() - 1 - i) / (float)(colorMap.size() - 1);
        float b = i / (float)(colorMap.size() - 1);
        colorMap[i].x = (uint8_t)(a * start.x + b * end.x);
        colorMap[i].y = (uint8_t)(a * start.y + b * end.y);
        colorMap[i].z = (uint8_t)(a * start.z + b * end.z);
        colorMap[i].w = (uint8_t)(a * start.w + b * end.w);
        // printf("i: %d.%d.%d.%d\n", colorMap[i].x, colorMap[i].y, colorMap[i].z, colorMap[i].w);
    }

    _depthColor.reset(Texture::create());
    DS3D_FAILED_RETURN(_depthColor, ErrCode::kGL, "Create depth color texture failed");
    DS3D_ERROR_RETURN(
        _depthColor->uploadImage(
            colorMap.size(), 1, colorMap.data(), GL_UNSIGNED_BYTE, GL_RGBA8, GL_RGBA),
        "upload depth color map failed.");

    _image.reset(Texture::create());
    DS3D_FAILED_RETURN(_image, ErrCode::kGL, "Create image texture failed");

    _depth.reset(Texture::create());
    DS3D_FAILED_RETURN(_depth, ErrCode::kGL, "Create depth texture failed");
    return ErrCode::kGood;
}

ErrCode
DepthRenderCtxImpl::uploadData(GuardDataMap datamap)
{
    Frame2DGuard depthFrame;
    DepthScale scale;

    if (datamap.hasData(kColorFrame)) {
        Frame2DGuard imageFrame;
        DS3D_ERROR_RETURN(
            datamap.getGuardData(kColorFrame, imageFrame),
            "No image found in datamap during video rendering");
        DS3D_ERROR_RETURN(uploadImage(imageFrame), "Upload color image into texture failed.");
    }

    if (datamap.hasData(kDepthFrame)) {
        Frame2DGuard depthFrame;
        DS3D_ERROR_RETURN(
            datamap.getGuardData(kDepthFrame, depthFrame),
            "No image found in datamap during video rendering");
        DS3D_ERROR_RETURN(uploadDepth(depthFrame), "Upload color image into texture failed.");

        DS3D_ERROR_RETURN(datamap.getData(kDepthScaleUnit, scale), "get depth scale failed.");
        _depthScale = scale.scaleUnit;
    }

    return ErrCode::kGood;
}

ErrCode
DepthRenderCtxImpl::uploadImage(Frame2DGuard& image)
{
    DS_ASSERT(image);
    const auto& plane = image->getPlane(0);
    _imageProgInfo.texAspectRatio = plane.width / (float)plane.height;

    DS3D_FAILED_RETURN(
        image->frameType() == FrameType::kColorRGBA, ErrCode::kUnsupported,
        "video render upload image only support RGBA color");

    auto preload = [this, &plane]() {
        // glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // this is just for RGB or R
        glPixelStorei(GL_UNPACK_ROW_LENGTH, plane.pitchInBytes / plane.bytesPerPixel);
    };

    auto postload = [this]() {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    };

    DS3D_ERROR_RETURN(
        _image->uploadImage(
            plane.width, plane.height, image->base(), GL_UNSIGNED_BYTE, GL_RGBA8, GL_RGBA, preload,
            postload),
        "upload depth color map failed.");

    return ErrCode::kGood;
}

ErrCode
DepthRenderCtxImpl::uploadDepth(Frame2DGuard& depth)
{
    DS_ASSERT(depth);
    const auto& plane = depth->getPlane(0);
    _depthProgInfo.texAspectRatio = plane.width / (float)plane.height;

    DS3D_FAILED_RETURN(
        depth->frameType() == FrameType::kDepth && depth->dataType() == DataType::kUint16,
        ErrCode::kUnsupported, "video render upload depth only support RGBA color");

    DS_ASSERT(plane.bytesPerPixel == 2);
    auto preload = [this, &plane]() {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, plane.pitchInBytes / plane.bytesPerPixel);
    };

    auto postload = [this]() {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    };

    DS3D_ERROR_RETURN(
        _depth->uploadImage(
            plane.width, plane.height, depth->base(), GL_UNSIGNED_BYTE, GL_RG8, GL_RG, preload,
            postload),
        "upload depth color map failed.");

    return ErrCode::kGood;
}

ErrCode
DepthRenderCtxImpl::updateLayout()
{
    DS_ASSERT(_window);
    uint32_t winW = 0, winH = 0;
    DS3D_ERROR_RETURN(_window->getFrameBufferSize(winW, winH), "get frame buffer size failed");

    if (_imageProgInfo.enabled && _depthProgInfo.enabled) {
        _imageProgInfo.x = 0;
        _imageProgInfo.y = 0;
        _imageProgInfo.width = winW / 2;
        _imageProgInfo.height = winH;

        _depthProgInfo.x = winW / 2;
        _depthProgInfo.y = 0;
        _depthProgInfo.width = winW / 2;
        _depthProgInfo.height = winH;
    } else {
        _imageProgInfo.x = 0;
        _imageProgInfo.y = 0;
        _imageProgInfo.width = winW;
        _imageProgInfo.height = winH;

        _depthProgInfo.x = 0;
        _depthProgInfo.y = 0;
        _depthProgInfo.width = winW;
        _depthProgInfo.height = winH;
    }

    return ErrCode::kGood;
}

namespace {
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>
calcViewport(uint32_t x, uint32_t y, uint32_t w, uint32_t h, float ratio)
{
    uint32_t oX, oY, oW, oH;
    if (h * ratio < w) {
        oW = (uint32_t)(h * ratio);
        oH = h;
    } else {
        oW = w;
        oH = (uint32_t)(w / ratio);
    }
    oX = x + (w - oW) / 2;
    oY = y + (h - oH) / 2;
    return std::make_tuple(oX, oY, oW, oH);
}
}  // namespace

ErrCode
DepthRenderCtxImpl::draw(GuardDataMap datamap)
{
    uint32_t x = 0, y = 0;
    size_t w = 0, h = 0;
    Program* p = nullptr;

    DS3D_ERROR_RETURN(updateLayout(), "update layout failed.");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (_imageProgInfo.enabled && datamap.hasData(kColorFrame)) {
        std::tie(x, y, w, h) = calcViewport(
            _imageProgInfo.x, _imageProgInfo.y, _imageProgInfo.width, _imageProgInfo.height,
            _imageProgInfo.texAspectRatio);
        p = imageProgram();

        DS3D_ERROR_RETURN(p->useProgram(), "use image program failed");
        glUniform1i(p->uniformLocation("imageTexture"), 0);

        glViewport(x, y, w, h);
        glScissor(x, y, w, h);
        glEnable(GL_SCISSOR_TEST);
        glDisable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT);

        DS_ASSERT(p);

        glActiveTexture(GL_TEXTURE0);
        _image->bind(GL_TEXTURE_2D);
        DS3D_RETURN_GL_ERROR("binding texture for render failed.");

        // DS3D_ERROR_RETURN(p->useProgram(), "use image program failed");
        _imageProgInfo.vao->bind();
        glDrawArrays(GL_TRIANGLES, 0, kVerTexNum);
        DS3D_RETURN_GL_ERROR("draw image texture failed.");
        _imageProgInfo.vao->unbind();
        _image->unbind(GL_TEXTURE_2D);
        glDisable(GL_SCISSOR_TEST);
        p->closeProgram();
    }

    if (_depthProgInfo.enabled && datamap.hasData(kDepthFrame)) {
        std::tie(x, y, w, h) = calcViewport(
            _depthProgInfo.x, _depthProgInfo.y, _depthProgInfo.width, _depthProgInfo.height,
            _depthProgInfo.texAspectRatio);
        p = depthProgram();

        DS_ASSERT(p);
        DS3D_ERROR_RETURN(p->useProgram(), "use depth program failed");

        glViewport(x, y, w, h);
        glScissor(x, y, w, h);
        glEnable(GL_SCISSOR_TEST);
        glDisable(GL_DEPTH_TEST);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUniform1f(p->uniformLocation("depthScale"), _depthScale);
        glUniform1f(p->uniformLocation("minDepth"), _minDepth);
        glUniform1f(p->uniformLocation("depthStep"), 1.0f / (_maxDepth - _minDepth));

        glUniform1i(p->uniformLocation("depthTexture"), 0);
        glUniform1i(p->uniformLocation("colorSampler"), 1);
        DS3D_RETURN_GL_ERROR("depth shader uniform locations failed");

        glActiveTexture(GL_TEXTURE0);
        _depth->bind(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE1);
        _depthColor->bind(GL_TEXTURE_2D);
        DS3D_RETURN_GL_ERROR("binding texture for render failed.");

        DS3D_ERROR_RETURN(p->useProgram(), "use depth program failed");
        _depthProgInfo.vao->bind();
        glDrawArrays(GL_TRIANGLES, 0, kVerTexNum);
        DS3D_RETURN_GL_ERROR("draw depth texture failed.");
        _depthProgInfo.vao->unbind();
        _depth->unbind(GL_TEXTURE_2D);
        glDisable(GL_SCISSOR_TEST);
        p->closeProgram();
    }

    return ErrCode::kGood;
}

ErrCode
DepthRenderCtxImpl::deinitResource()
{
    _window->close();
    auto freeProg = [](ProgramInfo& p) {
        p.vao.reset();
        p.program.reset();
    };
    freeProg(_imageProgInfo);
    freeProg(_depthProgInfo);
    _vertexBuf.reset();
    _coordBuf.reset();
    _image.reset();
    _depth.reset();
    _depthColor.reset();
    return ErrCode::kGood;
}

}}  // namespace ds3d::gl

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
gl::DepthRender*
NvDs3D_CreateVideoDepthRender(bool enableVideo, bool enableDepth)
{
    return new gl::DepthRenderCtxImpl(enableVideo, enableDepth);
}
DS3D_EXTERN_C_END