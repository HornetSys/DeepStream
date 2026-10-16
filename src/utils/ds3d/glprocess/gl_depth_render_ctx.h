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

#ifndef DS3D_GLPROCESS_GL_DEPTH_RENDER_CTX_H
#define DS3D_GLPROCESS_GL_DEPTH_RENDER_CTX_H

#include "ds3d/common/hpp/frame.hpp"

#include "gl_render_ctx.h"
#include "glfw_init.h"
#include "gl_program.h"
#include "gl_buf.h"

namespace ds3d { namespace gl {

class DepthRenderCtxImpl : public DepthRender {
public:
    DepthRenderCtxImpl(bool enableVideo, bool enableDepth)
    {
        _imageProgInfo.enabled = enableVideo;
        _depthProgInfo.enabled = enableDepth;
    }
    void setDepthRange(float min, float max) override
    {
        _minDepth = min;
        _maxDepth = max;
    }

    void setDepthColor(const vec3b& minDepthColor, const vec3b& maxDepthColor) override
    {
        _minDepthColor = minDepthColor;
        _maxDepthColor = maxDepthColor;
    }

    ~DepthRenderCtxImpl() = default;

    struct ProgramInfo {
        bool enabled = true;
        std::unique_ptr<Program> program;
        std::unique_ptr<VertexArray> vao;
        int inVertAttrib = -1;
        int inTexCoordAttrib = -1;
        uint32_t x = 0, y = 0, width = 0, height = 0;
        float texAspectRatio = 1.0f;
    };

    ErrCode initResource() override;
    ErrCode uploadData(GuardDataMap datamap) override;
    ErrCode draw(GuardDataMap datamap) override;
    ErrCode deinitResource() override;

private:
    Program* imageProgram() const { return _imageProgInfo.program.get(); }
    Program* depthProgram() const { return _depthProgInfo.program.get(); }
    WindowInfo _initWindowInfo{1280, 720, ""};

    ErrCode initVertexBuffer();
    ErrCode initVAO();
    ErrCode initTextures();

    ErrCode uploadImage(Frame2DGuard& image);
    ErrCode uploadDepth(Frame2DGuard& depth);

    ErrCode updateLayout();

    std::unique_ptr<VertexBuf> _vertexBuf;
    std::unique_ptr<VertexBuf> _coordBuf;

    ProgramInfo _imageProgInfo;
    ProgramInfo _depthProgInfo;

    std::unique_ptr<Texture> _image;
    std::unique_ptr<Texture> _depth;
    std::unique_ptr<Texture> _depthColor;
    float _depthScale = 1.0f;
    float _minDepth = 0.3f;  // meters
    float _maxDepth = 5.0f;  // meters
    vec3b _minDepthColor = {{200, 200, 0}};
    vec3b _maxDepthColor = {{50, 50, 50}};
};

}}  // namespace ds3d::gl

#endif
