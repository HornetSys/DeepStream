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

#ifndef DS3D_GLPROCESS_GL_RENDER_CTX_H
#define DS3D_GLPROCESS_GL_RENDER_CTX_H

#include "ds3d/common/common.h"
#include "ds3d/common/hpp/datamap.hpp"
#include "ds3d/common/hpp/frame.hpp"
#include "ds3d/glprocess/window.h"

namespace ds3d { namespace gl {

class iGLRender {
public:
    iGLRender() = default;
    virtual ~iGLRender() = default;

    virtual void setWindow(GuardGlWindow w) = 0;
    virtual GuardGlWindow getWindow() = 0;
    virtual ErrCode initResource() = 0;
    virtual ErrCode uploadData(GuardDataMap datamap) = 0;
    virtual ErrCode syncData() = 0;
    virtual ErrCode draw(GuardDataMap datamap) = 0;
    virtual ErrCode deinitResource() = 0;

    // add another interface to decouple renders and make it scalable
    // <configure> should be called before <initResource>
    virtual ErrCode configure(const char* yamlConfig, uint32_t size) { return ErrCode::kGood; }
    // flush temporary data
    virtual ErrCode flush() { return ErrCode::kGood; }
};

struct WindowInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    std::string title;
};

class BaseRender : public iGLRender {
public:
    void setWindow(GuardGlWindow w) override { _window = std::move(w); }
    GuardGlWindow getWindow() override { return _window; }
    ErrCode syncData() override { return ErrCode::kGood; }

protected:
    GuardGlWindow _window;
};

class DepthRender : public BaseRender {
public:
    virtual void setDepthRange(float min, float max) = 0;
    virtual void setDepthColor(const vec3b& minDepthColor, const vec3b& maxDepthColor) = 0;
};

class PointCloudRender : public BaseRender {
public:
    virtual void setModelMat(const std::vector<vec4f>& mat) = 0;
    virtual void setViewMat(const std::vector<vec4f>& mat) = 0;
    virtual void setLookAt(const vec3f& positon, const vec3f& target, const vec3f& up) = 0;
    virtual void setPerspectiveMat(const std::vector<vec4f>& mat) = 0;
    virtual void setPerspective(float degree, float ratio, float near, float far) = 0;
    virtual void setTexCoordY(bool opposite) = 0;
    virtual void setPositiveZOnly(bool enable) = 0;
    virtual void setElementSize(uint32_t elements) = 0;
    virtual void setPointsDataKey(const std::string& key) {}
    virtual void set3DBboxKey(const std::string& key) {}
    virtual void setLidarDataColor(const vec3b& color) {}
    virtual void setMaxPoints(uint32_t num) {}
    virtual void setMaxBboxes(uint32_t num) {}
    virtual void enableLabels(bool flag) {}
};

}}  // namespace ds3d::gl

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API ds3d::gl::DepthRender* NvDs3D_CreateVideoDepthRender(
    bool enableVideo, bool enableDepth);
DS3D_EXPORT_API ds3d::gl::PointCloudRender* NvDs3D_CreatePointCloudRender();
DS3D_EXPORT_API ds3d::gl::PointCloudRender* NvDs3D_CreateLidarRender();
DS3D_EXPORT_API ds3d::gl::iGLRender* NvDs3D_CreateTexture3DRender(
    const char* yamlConfig, uint32_t size);

DS3D_EXTERN_C_END

#endif  // DS3D_GLPROCESS_GL_RENDER_CTX_H