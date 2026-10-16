/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DS3D_GLPROCESS_GL_LIDAR_RENDER_CTX_H
#define DS3D_GLPROCESS_GL_LIDAR_RENDER_CTX_H

#include "ds3d/common/common.h"
#include "ds3d/common/hpp/datamap.hpp"
#include "ds3d/common/hpp/frame.hpp"

#include "glprocess/gl_program.h"
#include "glprocess/gl_buf.h"
#include "glprocess/window.h"
#include "glprocess/gl_render_ctx.h"
#include "glprocess/gl_text.h"
#include "glprocess/gl_point_cloud_render_ctx.h"
#include "ds3d/glprocess/gl_cuda_utils.h"

namespace ds3d { namespace gl {


static constexpr const uint32_t kDefaultMaxPoints = 2048000;
static constexpr const uint32_t kDefaultMax3DBboxes = 2048;

struct LidarRenderConfig : public BaseRenderConfig {
    uint32_t elementSize = 4;
    std::string lidarDataKey = kLidarXYZI;
    std::string lidarBboxKey = kLidar3DBboxRawData;
    vec3b lidarColor = {{0, 255, 0}};  // RGB
    bool enableLabel = false;
    // projecting lidar to image need image width/height settings
    bool enableLidarTo2DImageProject = false;
    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;
    std::string intrinsicsMatKey;
    std::string extrinsicsMatKey;
    // render range in xyz axes, default range [-2^20, - 2^20]
    vec2f xyzRange[3] = {
        {{-1048576.0f, 1048576.0f}}, {{-1048576.0f, 1048576.0f}}, {{-1048576.0f, 1048576.0f}}};
    bool xRangeEnabled = false;
    bool yRangeEnabled = false;
    bool zRangeEnabled = false;
    float lineWidth = 1.0f;
    uint32_t fontSize = kTextDefaultFontSize;
};

class LidarRenderCtxImpl : public PointCloudRenderCtxImpl {
public:
    LidarRenderCtxImpl();
    ~LidarRenderCtxImpl() = default;

    ErrCode initResource() override;
    ErrCode uploadData(GuardDataMap datamap) override;
    ErrCode draw(GuardDataMap datamap) override;
    ErrCode deinitResource() override;
    void setElementSize(uint32_t elements) override
    {
        DS_ASSERT(elements == 3 || elements == 4);
        _elementSize = elements;
    }
    void setPointsDataKey(const std::string& key) override { _lidarDataName = key; }
    void set3DBboxKey(const std::string& key) override { _lidarBboxName = key; }

    void setLidarDataColor(const vec3b& color) override
    {
        _pointColor = {{color.x / 255.0f, color.y / 255.0f, color.z / 255.0f, 1.0f}};
    }

    void setMaxPoints(uint32_t num) override { _maxPoints = num; }
    void setMaxBboxes(uint32_t num) override { _maxBboxes = num; }
    void enableLabels(bool flag) override { _enableLabels = flag; }

private:
    ErrCode configure(const char* yamlConfig, uint32_t size) override;
    ErrCode initVertexBuffer();
    ErrCode initVAO();
    ErrCode initTextResource();
    uint32_t elementSize() const { return _elementSize; }

    ErrCode uploadLidarData(FrameGuard& lidarData);
    ErrCode uploadLidarBboxData(FrameGuard& lidarRawBbox);
    ErrCode uploadTransformMat(GuardDataMap datamap);
    ErrCode generateBboxBufs(
        const Lidar3DBbox* bbox, uint32_t num, std::vector<vec3f>& boxBuf,
        std::vector<vec4f>& colorBuf);
    ErrCode uploadLabelImageBufs(const Lidar3DBbox* bbox, uint32_t num);
    ErrCode drawLabels();
    mat4f getMvp();

    LidarRenderConfig _conf;
    NormalViewPort _normalizedView;
    bool _confParsed = false;

    std::unique_ptr<Program> _pointProgram;
    std::unique_ptr<Program> _bboxProgram;
    std::unique_ptr<Program> _textProgram;
    Ptr<VertexBuf> _pointsBuf;
    std::unique_ptr<VertexArray> _pointsVao;
    Ptr<CudaVertexBuf> _cuPointsGlBuf;

    std::unique_ptr<VertexBuf> _bboxVertBuf;
    std::unique_ptr<VertexBuf> _bboxColorBuf;
    std::unique_ptr<VertexArray> _bboxVao;

    std::unique_ptr<VertexBuf> _textVertexBuf;
    std::unique_ptr<VertexBuf> _textUvBuf;
    std::unique_ptr<VertexArray> _textVao;
    std::vector<std::unique_ptr<Texture>> _textImages;
    uint32_t _labelCount = 0;

    bool _enableLabels = false;
    int _pointAttrib = -1;
    uint32_t _maxPoints = kDefaultMaxPoints;
    uint32_t _maxBboxes = kDefaultMax3DBboxes;
    vec4f _pointColor{{0, 1.0, 0, 1.0f}};
    uint32_t _elementSize = 4;  // XYZI by default
    std::string _lidarDataName = kLidarXYZI;
    std::string _lidarBboxName = kLidar3DBboxRawData;

    mat4f _frameExtrinsicMat = eye4x4();
    mat4f _frameIntrisicMat = eye4x4();
};

}}  // namespace ds3d::gl


#endif  // DS3D_GLPROCESS_GL_LIDAR_RENDER_CTX_H