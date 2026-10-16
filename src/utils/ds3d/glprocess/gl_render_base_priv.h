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

#ifndef DS3D_GLPROCESS_GL_RENDER_BASE_PRIV_H
#define DS3D_GLPROCESS_GL_RENDER_BASE_PRIV_H

#include "ds3d/common/common.h"
#include "ds3d/common/hpp/datamap.hpp"
#include "ds3d/common/helper/cuda_utils.h"
#include "ds3d/glprocess/gl_program.h"
#include "ds3d/glprocess/gl_buf.h"
#include "ds3d/glprocess/gl_render_ctx.h"
#include "ds3d/glprocess/gl_cuda_utils.h"

namespace ds3d { namespace gl {

/// @brief Normalized view port positon in the window in range [0, 1]
struct NormalViewPort {
    /// lower left in viewpoint coordinates
    vec2f xy0 = {{0, 0}};
    /// upper right in viewpoint coordinates
    vec2f xy1 = {{0, 0}};
};

/// @brief layout position in window coordinates
struct ViewLayout {
    /// top left in window coordinates
    vec2<int32_t> xy0 = {{0, 0}};
    /// buttom right in window coordinates
    vec2<int32_t> xy1= {{0, 0}};
};

struct ViewLookat {
    vec3f position{{0.0f, 0.0f, 1.0f}};
    vec3f target{{0.0f, 0.0f, -1.0f}};
    vec3f up{{0.0f, 1.0f, 0.0f}};
};

struct PerspectiveParam {
    float fov = 90.0f;  // degree
    float ratio = 0.0f;  // width / height set by default
    float near = 0.01f;
    float far = 100.0f;
};

/**
 * @brief base config for all renders
 * layout: [0, 0, 1280, 720]
 * color_clear: True
 */
struct BaseRenderConfig {
    ViewLayout viewLayout;
    ViewLookat lookat;
    bool lookatEnabled = false;
    PerspectiveParam perspective;
    bool perspectiveEnabled = false;
    mat4f modelMat = eye4x4();
    bool needColorClear = true;
    bool is2dOnly = false;
    int devId = 0;
};

class BaseRenderPriv : public BaseRender {
protected:
    Ptr<Program> _shader;
    Ptr<VertexArray> _vao;
    Ptr<CudaStream> _cudaStream;

    /// normalized viewport posion in the window
    NormalViewPort _normalizedView;
    mat4f _modelMat = eye4x4();
    mat4f _viewlMat = eye4x4();
    mat4f _projMat = eye4x4();

public:
    BaseRenderPriv() = default;
    virtual ~BaseRenderPriv() = default;
    ErrCode syncData() override
    {
        if (_cudaStream) {
            DS3D_ERROR_RETURN(_cudaStream->sync(), "BaseRenderPriv sync cuda stream failed");
        }
        return ErrCode::kGood;
    }

protected:
    Ptr<Program>& shader() { return _shader; }
    Ptr<VertexArray>& vao() { return _vao; }
    NormalViewPort& normalizedView() { return _normalizedView; }
    mat4f& model() { return _modelMat; }
    mat4f& view() { return _viewlMat; }
    mat4f& project() { return _projMat; }

    ErrCode initBase(const BaseRenderConfig& conf)
    {
        _cudaStream.reset(new CudaStream(cudaStreamDefault, conf.devId));
        DS3D_FAILED_RETURN(
            _cudaStream, ErrCode::kConfig, "BaseRenderPriv create cuda stream failed");
        return ErrCode::kGood;
    }

    ErrCode clearBase()
    {
        if (_cudaStream) {
            // skip error check for deinit
            _cudaStream->sync();
            _cudaStream.reset();
        }
        return ErrCode::kGood;
    }

    void initMvpFromConfig(const BaseRenderConfig& conf)
    {
        getMVPFrom(_modelMat, _viewlMat, _projMat, conf);
    }

    ErrCode buildShader(const std::vector<std::string>& scripts, const std::vector<GLenum>& types)
    {
        std::unique_ptr<Program> shader(BuildProgramFromSource(scripts, types));
        DS3D_FAILED_RETURN(shader, ErrCode::kGL, "Build shaders failed");
        _shader = std::move(shader);
        return ErrCode::kGood;
    }

    ErrCode storeViewPort(const ViewLayout& layout)
    {
        GuardGlWindow win = getWindow();
        DS_ASSERT(win);
        uint32_t winW = 0, winH = 0;
        DS3D_ERROR_RETURN(win->getWindowSize(winW, winH), "get frame buffer size failed");
        DS3D_FAILED_RETURN(winW && winH, ErrCode::kParam, "window size is 0");
        normalizeViewPort(layout, winW, winH, _normalizedView);
        return ErrCode::kGood;
    }

    ErrCode calculateViewPort(int32_t& x, int32_t& y, uint32_t& width, uint32_t& height)
    {
        GuardGlWindow win = getWindow();
        DS_ASSERT(win);
        uint32_t winW = 0, winH = 0;
        DS3D_ERROR_RETURN(win->getFrameBufferSize(winW, winH), "get frame buffer size failed");
        DS3D_FAILED_RETURN(winW && winH, ErrCode::kParam, "window size is 0");
        calculateViewPort(_normalizedView, winW, winH, x, y, width, height);

        return ErrCode::kGood;
    }

    ErrCode createVao()
    {
        _vao.reset(VertexArray::create());
        return ErrCode::kGood;
    }

public:
    /// static functions for backward compatibility
    static ErrCode parseConfig(const std::string& yamlStr, BaseRenderConfig& config)
    {
        YAML::Node node = YAML::Load(yamlStr);
        auto yLayout = node["layout"];
        auto ycolorClear = node["color_clear"];

        auto yViewPos = node["view_position"];
        auto yViewTarget = node["view_target"];
        auto yViewUp = node["view_up"];
        auto yNear = node["perspective_near"];
        auto yFar = node["perspective_far"];
        auto yFov = node["perspective_fov"];
        auto yRatio = node["perspective_ratio"];
        auto yModelMat = node["model_mat"];

        if (yLayout) {
            auto vec = yLayout.as<std::vector<int32_t>>();
            DS3D_FAILED_RETURN(
                vec.size() == 4, ErrCode::kConfig, "layout must be list of [x0, y0, x1, y1]");
            config.viewLayout.xy0 = {{vec[0], vec[1]}};
            config.viewLayout.xy1 = {{vec[2], vec[3]}};
            DS3D_FAILED_RETURN(
                config.viewLayout.xy0.x <= config.viewLayout.xy1.x &&
                config.viewLayout.xy0.y <= config.viewLayout.xy1.y,
                ErrCode::kConfig, "layout xy1 must >= xy0 [x0, y0, x1, y1]");
        }
        if (ycolorClear) {
            config.needColorClear = ycolorClear.as<bool>();
        }
        if (yViewPos) {
            auto pos = yViewPos.as<std::vector<float>>();
            DS3D_FAILED_RETURN(
                pos.size() >= 3, ErrCode::kConfig, "view_position need x,y,z 3 values");
            config.lookat.position = {{pos[0], pos[1], pos[2]}};
            config.lookatEnabled = true;
        }
        if (yViewTarget) {
            auto target = yViewTarget.as<std::vector<float>>();
            DS3D_FAILED_RETURN(
                target.size() >= 3, ErrCode::kConfig, "view_target need x,y,z 3 values");
            config.lookat.target = {{target[0], target[1], target[2]}};
            config.lookatEnabled = true;
        }
        if (yViewUp) {
            auto up = yViewUp.as<std::vector<float>>();
            DS3D_FAILED_RETURN(up.size() >= 3, ErrCode::kConfig, "view_up need x,y,z 3 values");
            config.lookat.up = {{up[0], up[1], up[2]}};
            config.lookatEnabled = true;
        }
        if (yNear) {
            config.perspective.near = yNear.as<float>();
            config.perspectiveEnabled = true;
        }
        if (yFar) {
            config.perspective.far = yFar.as<float>();
            config.perspectiveEnabled = true;
        }
        if (yFov) {
            config.perspective.fov = yFov.as<float>();
            config.perspectiveEnabled = true;
        }
        if (yRatio) {
            config.perspective.ratio = yRatio.as<float>();
            config.perspectiveEnabled = true;
        }

        if (config.perspectiveEnabled && isNear(config.perspective.ratio, 0.0f)) {
            int h = config.viewLayout.xy1.y - config.viewLayout.xy0.y;
            int w = config.viewLayout.xy1.x - config.viewLayout.xy0.x;
            if (h > 0) {
                config.perspective.ratio = (float)(w) / (float)(h);
            } else {
                config.perspective.ratio = 1.0f;
            }
        }

        if (yModelMat) {
            auto m4x4 = yModelMat.as<std::vector<float>>();
            DS3D_FAILED_RETURN(
                m4x4.size() == 16, ErrCode::kConfig,
                "model_mat must be a list of 16(4x4) float values");
            config.modelMat = eye4x4();
            for (int r = 0; r < 4; ++r) {
                config.modelMat[r] = {
                    {m4x4[r * 4], m4x4[r * 4 + 1], m4x4[r * 4 + 2], m4x4[r * 4 + 3]}};
            }
        }
        return ErrCode::kGood;
    }

    static void getMVPFrom(mat4f& m, mat4f& v, mat4f& p, const BaseRenderConfig& config)
    {
        v = eye4x4();
        p = eye4x4();
        if (!config.is2dOnly && config.lookatEnabled) {
            glm::mat4 mat = glm::lookAt(
                vec3ToGlm(config.lookat.position), vec3ToGlm(config.lookat.target),
                vec3ToGlm(config.lookat.up));
            v = glmMat4ToDS(mat);
        }
        if (!config.is2dOnly && config.perspectiveEnabled) {
            glm::mat4 mat = glm::perspective(
                glm::radians(config.perspective.fov), config.perspective.ratio,
                config.perspective.near, config.perspective.far);
            p = glmMat4ToDS(mat);
        }
        m = config.modelMat;
    }

    static void normalizeViewPort(
        const ViewLayout& layout, uint32_t winW, uint32_t winH,
        NormalViewPort& normalizedView)
    {
        DS_ASSERT(winW && winH);
        normalizedView.xy0.x = layout.xy0.x / (float)winW;
        normalizedView.xy0.y = 1.0 - layout.xy1.y / (float)winH;
        normalizedView.xy1.x = layout.xy1.x / (float)winW;
        normalizedView.xy1.y = 1.0 - layout.xy0.y / (float)winH;
    }

    static void calculateViewPort(
        const NormalViewPort& normalizedView, uint32_t winW, uint32_t winH, int32_t& x, int32_t& y,
        uint32_t& width, uint32_t& height)
    {
        x = (int32_t)(normalizedView.xy0.x * winW);
        y = (int32_t)(normalizedView.xy0.y * winH);

        int32_t w = (int32_t)(normalizedView.xy1.x * winW) - x;
        int32_t h = (int32_t)(normalizedView.xy1.y * winH) - y;
        w = std::max<int32_t>(0, w);
        h = std::max<int32_t>(0, h);
        width = (uint32_t)w;
        height = (uint32_t)h;
    }
};


}}  // namespace ds3d::gl

#endif  // DS3D_GLPROCESS_GL_RENDER_BASE_PRIV_H