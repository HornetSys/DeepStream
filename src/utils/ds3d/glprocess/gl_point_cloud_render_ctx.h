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

#ifndef DS3D_GLPROCESS_GL_POINTCLOUD_RENDER_CTX_H
#define DS3D_GLPROCESS_GL_POINTCLOUD_RENDER_CTX_H

#include "ds3d/common/common.h"
#include "ds3d/common/hpp/datamap.hpp"
#include "ds3d/common/hpp/frame.hpp"
#include "ds3d/common/helper/cuda_utils.h"

#include "glprocess/gl_program.h"
#include "glprocess/gl_buf.h"
#include "glprocess/window.h"
#include "glprocess/gl_render_ctx.h"
#include "glprocess/gl_render_base_priv.h"

namespace ds3d { namespace gl {

constexpr uint32_t kMaxPointsGL = 1920 * 1080;

struct PointCloudConfig : public BaseRenderConfig {
    bool coordYOpposite = true;
    bool positiveZOnly = false;
};

class PointCloudRenderCtxImpl : public PointCloudRender {
public:
    PointCloudRenderCtxImpl();
    ~PointCloudRenderCtxImpl() = default;
    // void setMaxPoints(uint32_t count) { _maxPoints = count; }
    void setModelMat(const std::vector<vec4f>& mat) override
    {
        DS_ASSERT(mat.size() == 4);
        _modelMat = mat;
    }
    void setViewMat(const std::vector<vec4f>& mat) override
    {
        DS_ASSERT(mat.size() == 4);
        _viewMat = mat;
    }
    void setLookAt(const vec3f& positon, const vec3f& target, const vec3f& up) override;
    void setPerspectiveMat(const std::vector<vec4f>& mat) override
    {
        DS_ASSERT(mat.size() == 4);
        _projMat = mat;
    }
    void setPerspective(float degree, float ratio, float near, float far) override;
    void setTexCoordY(bool opposite) override { _texCoordYOpposite = opposite; }
    void setPositiveZOnly(bool enable) override { _enablePositiveZOnly = enable; }
    void setElementSize(uint32_t elements)
    {
        DS_ASSERT(elements == 3);
        DS3D_UNUSED(elements);
    }

protected:
    ErrCode configure(const char* yamlConfig, uint32_t size) override;
    ErrCode initResource() override;
    ErrCode uploadData(GuardDataMap datamap) override;
    ErrCode syncData() override;
    ErrCode draw(GuardDataMap datamap) override;
    ErrCode deinitResource() override;
    virtual ErrCode initMouseCallback(GuardGlWindow& window);

private:
    ErrCode initVertexBuffer();
    ErrCode initVAO();
    ErrCode initTextures();

    ErrCode uploadImage(Frame2DGuard& image);
    ErrCode uploadPoints(FrameGuard& points, FrameGuard& colorCoord);

private:
    PointCloudConfig _conf;
    NormalViewPort _normalizedView;
    bool _confParsed = false;

protected:
    WindowInfo _initWindowInfo{1280, 720, ""};
    ViewLookat _viewLookat;
    float _lastX = 0;
    float _lastY = 0;
    float _mouseMoveYaw = 0;
    bool _useViewLookat = false;
    std::vector<vec4f> _modelMat;  // 4x4 column-major
    std::vector<vec4f> _viewMat;  // 4x4 column-major
    std::vector<vec4f> _projMat;  // 4x4 column-major

    Ptr<CudaStream> _cudaStream;

private:
    std::unique_ptr<VertexBuf> _vertexBuf;
    std::unique_ptr<VertexBuf> _coordBuf;
    uint32_t _maxPoints = kMaxPointsGL;
    std::unique_ptr<Texture> _colorImage;

    std::unique_ptr<Program> _program;
    std::unique_ptr<VertexArray> _vao;
    int _inVertAttrib = -1;
    int _inTexCoordAttrib = -1;
    bool _texCoordYOpposite = true;
    bool _enablePositiveZOnly = false;
};

}}  // namespace ds3d::gl


#endif  // DS3D_GLPROCESS_GL_POINTCLOUD_RENDER_CTX_H