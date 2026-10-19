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
#include "ds3d/glprocess/gl_helper.h"
#include "ds3d/common/ds3d_analysis_datatype.h"
#include "ds3d/common/hpp/yaml_config.hpp"

#include "ds3d/glprocess/gl_lidar_render_ctx.h"
#include "ds3d/glprocess/glfw_init.h"
#include "ds3d/glprocess/gl_helper.h"
#include "ds3d/glprocess/gl_script.h"
#include "ds3d/glprocess/gl_text.h"

#include "glm/ext.hpp"
#include "glm/gtx/quaternion.hpp"

#define BBOX_3D_VERTEX_NUM 24
#define LABEL_VERTEX_NUM 6

namespace ds3d { namespace gl {

static const float kTextScaleX = 1.0;
static const float kTextScaleY = 1.0;

LidarRenderCtxImpl::LidarRenderCtxImpl() {}

ErrCode
LidarRenderCtxImpl::configure(const char* yamlConfig, uint32_t size)
{
    LidarRenderConfig conf;
    std::string yamlStr = cppString(yamlConfig, size);
    DS3D_ERROR_RETURN(
        BaseRenderPriv::parseConfig(yamlStr, conf), "parse lidar base config failed.");
    YAML::Node node = YAML::Load(yamlStr);
    auto yEleSize = node["element_size"];
    auto yLidarDataKey = node["lidar_data_key"];
    auto yLidarBboxKey = node["lidar_bbox_key"];
    auto yColor = node["lidar_color"];
    auto ylabel = node["enable_label"];
    auto yPrject2Image = node["project_lidar_to_image"];
    auto yImgW = node["image_width"];
    auto yImgH = node["image_height"];
    auto yIntrMatKey = node["intrinsics_mat_key"];
    auto yExtrMatKey = node["extrinsics_mat_key"];
    auto yZRange = node["z_range"];
    auto yYRange = node["y_range"];
    auto yXRange = node["x_range"];
    auto yLineWidth = node["line_width"];
    auto yFontSize = node["font_size"];

    if (yEleSize) {
        conf.elementSize = yEleSize.as<uint32_t>();
    }
    if (yLidarDataKey) {
        conf.lidarDataKey = yLidarDataKey.as<std::string>();
    }
    if (yLidarBboxKey) {
        conf.lidarBboxKey = yLidarBboxKey.as<std::string>();
    }
    if (yColor) {
        auto color = yColor.as<std::vector<uint32_t>>();
        size_t size = (color.size() < 3 ? color.size() : 3);
        for (size_t i = 0; i < size; ++i) {
            conf.lidarColor.data[i] = (uint8_t)color[i];
        }
    }
    if (ylabel) {
        conf.enableLabel = ylabel.as<bool>();
    }
    if (yPrject2Image) {
        conf.enableLidarTo2DImageProject = yPrject2Image.as<bool>();
    }
    if (yImgW) {
        conf.imageWidth = yImgW.as<uint32_t>();
    }
    if (yImgH) {
        conf.imageHeight = yImgH.as<uint32_t>();
    }
    if (yIntrMatKey) {
        conf.intrinsicsMatKey = yIntrMatKey.as<std::string>();
    }
    if (yExtrMatKey) {
        conf.extrinsicsMatKey = yExtrMatKey.as<std::string>();
    }
    if (yZRange) {
        auto range = yZRange.as<std::vector<float>>();
        DS3D_FAILED_RETURN(
            range.size() == 2, ErrCode::kConfig, "z_range must be list of 2 as [min, max]");
        conf.xyzRange[2] = {{range[0], range[1]}};
        conf.zRangeEnabled = true;
    }
    if (yYRange) {
        auto range = yYRange.as<std::vector<float>>();
        DS3D_FAILED_RETURN(
            range.size() == 2, ErrCode::kConfig, "y_range must be list of 2 as [min, max]");
        conf.xyzRange[1] = {{range[0], range[1]}};
        conf.yRangeEnabled = true;
    }
    if (yXRange) {
        auto range = yXRange.as<std::vector<float>>();
        DS3D_FAILED_RETURN(
            range.size() == 2, ErrCode::kConfig, "x_range must be list of 2 as [min, max]");
        conf.xyzRange[0] = {{range[0], range[1]}};
        conf.xRangeEnabled = true;
    }
    if (yLineWidth) {
        conf.lineWidth = yLineWidth.as<float>();
    }
    if (yFontSize) {
        conf.fontSize = yFontSize.as<uint32_t>();
    }

    if (conf.enableLidarTo2DImageProject) {
        if (conf.intrinsicsMatKey.empty()) {
            LOG_WARNING("lidar_to_image_project requires to set intrinsinc_mat_key");
        }
        DS3D_FAILED_RETURN(
            conf.imageWidth && conf.imageHeight, ErrCode::kConfig,
            "project_lidar_to_image requires image_width and image_height settings");
    }

    _conf = conf;
    _confParsed = true;

    if (conf.lookatEnabled) {
        setLookAt(conf.lookat.position, conf.lookat.target, conf.lookat.up);
    }
    if (conf.perspectiveEnabled && !conf.is2dOnly && !conf.enableLidarTo2DImageProject) {
        setPerspective(
            conf.perspective.fov, conf.perspective.ratio, conf.perspective.near,
            conf.perspective.far);
    }
    setElementSize(conf.elementSize);
    setPointsDataKey(conf.lidarDataKey);
    set3DBboxKey(conf.lidarBboxKey);
    setLidarDataColor(conf.lidarColor);
    enableLabels(conf.enableLabel);
    return ErrCode::kGood;
}

ErrCode
LidarRenderCtxImpl::initResource()
{
    if (!_window) {
        _window.reset(NvDs3D_CreateGlfwWindow(1280, 720, "point-cloud-render"));
        DS3D_FAILED_RETURN(
            _window, ErrCode::kGL, "create glfw window failed for resource creation");
    }

    DS_ASSERT(_window);
    _window->getWindowSize(_initWindowInfo.width, _initWindowInfo.height);
    _initWindowInfo.title = _window->getTitle();
    _lastX = _initWindowInfo.width / 2;
    _lastY = _initWindowInfo.height / 2;

    if (_confParsed) {
        _lastX = (_conf.viewLayout.xy0.x + _conf.viewLayout.xy1.x) / 2;
        _lastY = (_conf.viewLayout.xy0.y + _conf.viewLayout.xy1.y) / 2;
        BaseRenderPriv::normalizeViewPort(
            _conf.viewLayout, _initWindowInfo.width, _initWindowInfo.height, _normalizedView);
    }

    DS3D_ERROR_RETURN(initMouseCallback(_window), "init Mouse Callback failed.");

    DS_ASSERT(elementSize() == 4 || elementSize() == 3);
    char pointType[1024] = {0};
    snprintf(
        pointType, sizeof(pointType) - 1, "#define PointType %s \n",
        (elementSize() == 4 ? "vec4" : "vec3"));

    std::string lidarVertScript = kLidarVertexScript;
    std::string otherPrep;
    if (_conf.xRangeEnabled) {
        otherPrep += "#define X_RANGE \n";
    }
    if (_conf.yRangeEnabled) {
        otherPrep += "#define Y_RANGE \n";
    }
    if (_conf.zRangeEnabled) {
        otherPrep += "#define Z_RANGE \n";
    }
    if (_conf.xRangeEnabled || _conf.yRangeEnabled || _conf.zRangeEnabled) {
        lidarVertScript = kLidarZRangeVertexScript;
    }

    std::string vertexSource = std::string(GLSL_VERSION) + pointType + otherPrep + lidarVertScript;
    std::string fragmentSource = std::string(GLSL_VERSION) + kLidarFragmentScript;

    std::unique_ptr<Program> program(BuildProgramFromSource(
        {vertexSource, fragmentSource}, {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER}));
    DS3D_FAILED_RETURN(program, ErrCode::kGL, "Build lidar shaders failed");
    _pointProgram = std::move(program);
    DS_ASSERT(_pointProgram);

    std::unique_ptr<Program> bboxProgram(BuildProgramFromSource(
        {std::string(GLSL_VERSION) + kLidarBboxVertexScript,
         std::string(GLSL_VERSION) + kLidarBboxFragmentScript},
        {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER}));
    DS3D_FAILED_RETURN(bboxProgram, ErrCode::kGL, "Build lidar bbox shaders failed");
    _bboxProgram = std::move(bboxProgram);
    DS_ASSERT(_bboxProgram);

    _cudaStream.reset(new CudaStream(cudaStreamDefault, _conf.devId));
    DS3D_FAILED_RETURN(_cudaStream, ErrCode::kConfig, "lidar render create cuda stream failed");

    DS3D_ERROR_RETURN(initVertexBuffer(), "init Vertex Buffers failed");
    DS3D_ERROR_RETURN(initVAO(), "init all VAO failed");

    if (_enableLabels) {
        DS3D_ERROR_RETURN(initTextResource(), "init GL text resource failed");
    }

    return ErrCode::kGood;
}

ErrCode
LidarRenderCtxImpl::initVertexBuffer()
{
    _pointsBuf.reset(VertexBuf::create(GL_ARRAY_BUFFER));
    _bboxVertBuf.reset(VertexBuf::create(GL_ARRAY_BUFFER));
    _bboxColorBuf.reset(VertexBuf::create(GL_ARRAY_BUFFER));

    if (elementSize() == 3) {
        DS3D_ERROR_RETURN(
            _pointsBuf->allocate<vec3f>(_maxPoints, GL_DYNAMIC_DRAW), "allocate vertex data failed");
    } else if (elementSize() == 4) {
        DS3D_ERROR_RETURN(
            _pointsBuf->allocate<vec4f>(_maxPoints, GL_DYNAMIC_DRAW), "allocate vertex data failed");
    } else {
        LOG_ERROR(
            "allocate point buffer failed since element_size: %d is not supported", elementSize());
        return ErrCode::kUnsupported;
    }
    DS3D_ERROR_RETURN(
        _bboxVertBuf->allocate<vec3f>(_maxBboxes * BBOX_3D_VERTEX_NUM, GL_DYNAMIC_DRAW),
        "allocate bbox vertex data failed");
    DS3D_ERROR_RETURN(
        _bboxColorBuf->allocate<vec4f>(_maxBboxes * BBOX_3D_VERTEX_NUM, GL_DYNAMIC_DRAW),
        "allocate bbox vertex color data failed");

    DS3D_RETURN_GL_ERROR("GL init vertex buffers failed.");

    return ErrCode::kGood;
}

ErrCode
LidarRenderCtxImpl::initVAO()
{
    if (!_pointsVao) {
        _pointsVao.reset(VertexArray::create());
        DS3D_FAILED_RETURN(_pointsVao, ErrCode::kGL, "Create lidar VAO failed");
    }
    DS_ASSERT(_pointsVao);

    DS_ASSERT(_pointProgram);
    DS3D_ERROR_RETURN(_pointProgram->useProgram(), "use lidar program failed");
    _pointAttrib = _pointProgram->attribLocation("point");

    _pointsVao->bind();
    _pointsBuf->bind();
    // stride: 0 for tightly packed
    glVertexAttribPointer(_pointAttrib, elementSize(), GL_FLOAT, GL_FALSE, 0, nullptr);
    _pointsVao->enableAttrib(_pointAttrib);
    _pointsBuf->unbind();
    DS3D_RETURN_GL_ERROR("enable inVert into vao failed.");
    _pointsVao->unbind();
    DS3D_RETURN_GL_ERROR("enable inTexCoord into vao failed.");
    _pointProgram->closeProgram();
    DS3D_RETURN_GL_ERROR("GL init VAO for points program failed.");

    // bbox program starts
    if (!_bboxVao) {
        _bboxVao.reset(VertexArray::create());
        DS3D_FAILED_RETURN(_bboxVao, ErrCode::kGL, "Create lidar bbox VAO failed");
    }
    DS_ASSERT(_bboxProgram);
    DS3D_ERROR_RETURN(_bboxProgram->useProgram(), "use lidar program failed");
    int vertAttrib = _bboxProgram->attribLocation("vert");
    int colorAttrib = _bboxProgram->attribLocation("color");

    _bboxVao->bind();
    _bboxVertBuf->bind();
    // stride: 0 for tightly packed
    glVertexAttribPointer(vertAttrib, 3, GL_FLOAT, GL_FALSE, sizeof(vec3f), nullptr);
    _bboxVao->enableAttrib(vertAttrib);
    _bboxVertBuf->unbind();
    DS3D_RETURN_GL_ERROR("enable inVert into vao failed.");
    // bbox color
    _bboxColorBuf->bind();
    // stride: 0 for tightly packed
    glVertexAttribPointer(colorAttrib, 4, GL_FLOAT, GL_TRUE, sizeof(vec4f), nullptr);
    _bboxVao->enableAttrib(colorAttrib);
    _bboxColorBuf->unbind();
    _bboxVao->unbind();
    DS3D_RETURN_GL_ERROR("enable bbox vert and color into vao failed.");
    _bboxProgram->closeProgram();
    DS3D_RETURN_GL_ERROR("GL init VAO for bbox program failed.");

    return ErrCode::kGood;
}

ErrCode
LidarRenderCtxImpl::initTextResource()
{
    std::unique_ptr<Program> textProgram(BuildProgramFromSource(
        {std::string(GLSL_VERSION) + kImageTextureVertexScript,
         std::string(GLSL_VERSION) + kImageTextureFragmentScript},
        {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER}));

    DS3D_FAILED_RETURN(textProgram, ErrCode::kGL, "Build lidar text shaders failed");
    DS_ASSERT(textProgram);
    DS3D_ERROR_RETURN(textProgram->useProgram(), "use text program failed");
    int vertAttrib = textProgram->attribLocation("inVert");
    int uvAttrib = textProgram->attribLocation("inTexCoord");

    _textVertexBuf.reset(VertexBuf::create(GL_ARRAY_BUFFER));
    _textUvBuf.reset(VertexBuf::create(GL_ARRAY_BUFFER));

    DS3D_ERROR_RETURN(
        _textVertexBuf->allocate<vec3f>(_maxBboxes * LABEL_VERTEX_NUM, GL_DYNAMIC_DRAW),
        "allocate label vertex data failed");
    DS3D_ERROR_RETURN(
        _textUvBuf->allocate<vec2f>(_maxBboxes * LABEL_VERTEX_NUM, GL_DYNAMIC_DRAW),
        "allocate label uv data failed");
    DS3D_RETURN_GL_ERROR("GL allocate label buffers failed.");

    if (!_textVao) {
        _textVao.reset(VertexArray::create());
        DS3D_FAILED_RETURN(_textVao, ErrCode::kGL, "Create text VAO failed");
    }
    _textVao->bind();
    _textVertexBuf->bind();
    glVertexAttribPointer(vertAttrib, 3, GL_FLOAT, GL_FALSE, sizeof(vec3f), (const void*)0);
    _textVao->enableAttrib(vertAttrib);
    _textVertexBuf->unbind();
    DS3D_RETURN_GL_ERROR("GL text vao enable vertex attribute failed.");

    _textUvBuf->bind();
    glVertexAttribPointer(uvAttrib, 2, GL_FLOAT, GL_FALSE, sizeof(vec2f), (const void*)0);
    _textVao->enableAttrib(uvAttrib);
    _textUvBuf->unbind();
    DS3D_RETURN_GL_ERROR("GL text vao enable uv attribute failed.");

    _textVao->unbind();

    textProgram->closeProgram();
    _textProgram = std::move(textProgram);
    return ErrCode::kGood;
}

ErrCode
LidarRenderCtxImpl::uploadData(GuardDataMap datamap)
{
    if (datamap.hasData(_lidarDataName)) {
        FrameGuard lidarFrame;
        DS3D_ERROR_RETURN(
            datamap.getGuardData(_lidarDataName, lidarFrame),
            "No data: %s found in datamap during lidar data rendering", _lidarDataName.c_str());

        DS3D_ERROR_RETURN(uploadLidarData(lidarFrame), "Upload lidar data into vertex buffer failed.");
    }

    if (_confParsed) {
        DS3D_ERROR_RETURN(
            uploadTransformMat(datamap), "Upload lidar data into vertex buffer failed.");
    }

    if (datamap.hasData(_lidarBboxName)) {
        FrameGuard lidar3dBboxData;
        DS3D_ERROR_RETURN(
            datamap.getGuardData(_lidarBboxName, lidar3dBboxData),
            "No 3d bbox data: %s found in datamap during lidar data rendering",
            _lidarBboxName.c_str());

        DS3D_ERROR_RETURN(
            uploadLidarBboxData(lidar3dBboxData), "Upload lidar data into vertex buffer failed.");
    }

    return ErrCode::kGood;
}

ErrCode
LidarRenderCtxImpl::uploadLidarData(FrameGuard& points)
{
    DS3D_FAILED_RETURN(
        (points->frameType() == FrameType::kPointXYZ ||
         points->frameType() == FrameType::kLidarXYZI) &&
            points->dataType() == DataType::kFp32,
        ErrCode::kUnsupported,
        "lidar render upload points only support PointXYZ|LidarXYZI with FP32");

    Shape pShape = points->shape();  // N x 3 | N x 4
    DS_ASSERT(pShape.numDims);
    // XYZ | XYZI
    uint32_t frameEleSize = pShape.d[pShape.numDims - 1];
    DS3D_FAILED_RETURN(
        frameEleSize == elementSize(), ErrCode::kParam,
        "frame's element size: %u is not equal to config's element_size: %u", frameEleSize,
        elementSize());
    size_t numPoints = ShapeSize(pShape) / frameEleSize;
    numPoints = std::min<size_t>(numPoints, _maxPoints);
    size_t uploadBytes = numPoints * elementSize() * sizeof(float);
    MemType memType = points->memType();
    DS3D_FAILED_RETURN(_pointsBuf, ErrCode::kGL, "Check vertexBuffer for points failed");

    if (!_cuPointsGlBuf && memType == MemType::kGpuCuda) {
        _cuPointsGlBuf.reset(new CudaVertexBuf);
        DS3D_ERROR_RETURN(
            _cuPointsGlBuf->initCuda(_pointsBuf, 0), "Failed to register points vetexbuf to cuda");
    }

    if (memType == MemType::kGpuCuda) {
        DS_ASSERT(_cuPointsGlBuf);
        DS_ASSERT(_cudaStream);
        auto cudaProc = [src = points->base(), uploadBytes, this](
                            void* dst, size_t maxBytes, cudaStream_t stream) {
            DS3D_CHECK_CUDA_ERROR(
                cudaMemcpyAsync(
                    dst, src, std::min<size_t>(uploadBytes, maxBytes), cudaMemcpyDeviceToDevice,
                    stream),
                return ErrCode::kCuda, "cudaMemcpy2DToArray failed");
            return ErrCode::kGood;
        };
        _cuPointsGlBuf->cudaProcess(cudaProc, _cudaStream->get());
        // // manually call syncData() to synchrnize cuda stream for performance
    } else {
        DS3D_ERROR_RETURN(
            _pointsBuf->uploadSubData(0, uploadBytes, points->base()),
            "upload vertex points data failed");
    }

    return ErrCode::kGood;
}

ErrCode
LidarRenderCtxImpl::uploadTransformMat(GuardDataMap datamap)
{
    std::string extKey = _conf.extrinsicsMatKey;
    std::string intKey = _conf.intrinsicsMatKey;
    if (!extKey.empty() && datamap.hasData(extKey)) {
        TransformMatrix extMat;
        DS3D_ERROR_RETURN(
            datamap.getData(extKey, extMat),
            "No extrinsic: %s found in datamap during lidar data rendering", extKey.c_str());
        _frameExtrinsicMat.assign(std::begin(extMat.matrix), std::end(extMat.matrix));
    }
    if (!intKey.empty() && datamap.hasData(intKey)) {
        TransformMatrix intMat;
        DS3D_ERROR_RETURN(
            datamap.getData(intKey, intMat),
            "No intrinsic: %s found in datamap during lidar data rendering", intKey.c_str());
        _frameIntrisicMat.assign(std::begin(intMat.matrix), std::end(intMat.matrix));
    }
    return ErrCode::kGood;
}

ErrCode
LidarRenderCtxImpl::uploadLidarBboxData(FrameGuard& lidarRawBbox)
{
    DS3D_FAILED_RETURN(
        lidarRawBbox->frameType() == FrameType::kCustom, ErrCode::kUnsupported,
        "lidar raw bbox frame type must be kCustom");

    Shape pShape = lidarRawBbox->shape();  // 1 x N x sizeof(Lidar3DBbox)
    DS_ASSERT(pShape.numDims);
    if (!ShapeSize(pShape)) {
        LOG_DEBUG("empty lidar 3D bbox found in render.");
        return ErrCode::kGood;
    }
    uint32_t perBboxBytes = pShape.d[pShape.numDims - 1];
    DS_ASSERT(perBboxBytes == sizeof(Lidar3DBbox));
    uint32_t numBbox = ShapeSize(pShape) / perBboxBytes;
    numBbox = std::min(numBbox, _maxBboxes);
    Lidar3DBbox* bbox = (Lidar3DBbox*)lidarRawBbox->base();
    std::vector<vec3f> boxBuf;
    std::vector<vec4f> colorBuf;
    DS3D_ERROR_RETURN(
        generateBboxBufs(bbox, numBbox, boxBuf, colorBuf), "generate 3D bbox buffer failed.");

    DS3D_FAILED_RETURN(_bboxVertBuf, ErrCode::kGL, "Check bbox vertexBuffer for points failed");
    DS3D_ERROR_RETURN(
        _bboxVertBuf->uploadSubData(0, boxBuf.size() * sizeof(vec3f), boxBuf.data()),
        "upload bbox vertex data failed");
    DS3D_FAILED_RETURN(_bboxColorBuf, ErrCode::kGL, "Check colorBuffer for points failed");
    DS3D_ERROR_RETURN(
        _bboxColorBuf->uploadSubData(0, colorBuf.size() * sizeof(vec4f), colorBuf.data()),
        "upload bbox color data failed");

    if (_enableLabels) {
        DS3D_ERROR_RETURN(
            uploadLabelImageBufs(bbox, numBbox), "generate label image buffer failed.");
    }

    return ErrCode::kGood;
}

ErrCode
LidarRenderCtxImpl::uploadLabelImageBufs(const Lidar3DBbox* bbox, uint32_t num)
{
    _textImages.clear();
    _labelCount = 0;
    std::vector<vec3f> labelVertex;
    std::vector<vec2f> labelUv;
    uint32_t winW = 0, winH = 0;
    _window->getFrameBufferSize(winW, winH);
    uint32_t viewW = winW;
    uint32_t viewh = winH;
    if (_confParsed) {
        int32_t x = 0, y = 0;
        BaseRenderPriv::calculateViewPort(_normalizedView, winW, winH, x, y, viewW, viewh);
    }

    for (uint32_t i = 0; i < num && _labelCount < _maxBboxes; ++i) {
        if (!strlen(bbox[i].labels))
            continue;

        float cosYaw = cos(bbox[i].yaw) / 2.0f, sinYaw = sin(bbox[i].yaw) / 2.0f;
        vec4f pos = {
            {(-bbox[i].dx) * cosYaw + bbox[i].dy * (-sinYaw) + bbox[i].centerX,
             (-bbox[i].dx) * sinYaw + bbox[i].dy * cosYaw + bbox[i].centerY,
             bbox[i].centerZ - bbox[i].dz / 2.0f, 1.0f}};

        auto mvp = getMvp();
        glm::mat4 m = glm::make_mat4<float>((const float*)&mvp[0]);
        glm::vec4 p = glm::make_vec4<float>((const float*)&pos);
        glm::vec4 p2D = m * p;
        if (p2D.w <= std::numeric_limits<float>::epsilon())
            continue;

        const std::string label(bbox[i].labels);
        vec4f color = bbox[i].bboxColor;
        auto texture = CreateTextTexture(label, kTextDefaultFont, _conf.fontSize, color);
        float w = kTextScaleX * texture->w() / viewW;
        float h = kTextScaleY * texture->h() / viewh;
        float hGap = 3 / (float)viewh;
        _textImages.emplace_back(std::move(texture));

        vec3f posBL{{p2D.x / p2D.w, p2D.y / p2D.w - hGap - h, p2D.z / p2D.w}};  // bottom left
        auto vertWrite = [&posBL, &labelVertex](float offsetX, float offsetY) {
            labelVertex.push_back(vec3f{{posBL.x + offsetX, posBL.y + offsetY, posBL.z}});
        };
        auto uvWrite = [&labelUv](float offsetX, float offsetY) {
            labelUv.push_back(vec2f{{offsetX, offsetY}});
        };
        vertWrite(0, 0);
        uvWrite(0, 0);
        vertWrite(w, 0);
        uvWrite(1, 0);
        vertWrite(w, h);
        uvWrite(1, 1);

        vertWrite(w, h);
        uvWrite(1, 1);
        vertWrite(0, h);
        uvWrite(0, 1);
        vertWrite(0, 0);
        uvWrite(0, 0);
        ++_labelCount;
    }
    DS_ASSERT(labelVertex.size() == labelUv.size());
    DS_ASSERT(labelVertex.size() == _textImages.size() * LABEL_VERTEX_NUM);

    DS3D_ERROR_RETURN(
        _textVertexBuf->uploadSubData(0, labelVertex.size() * sizeof(vec3f), labelVertex.data()),
        "upload label vertex data failed");
    DS3D_ERROR_RETURN(
        _textUvBuf->uploadSubData(0, labelUv.size() * sizeof(vec2f), labelUv.data()),
        "upload label uv data failed");

    return ErrCode::kGood;
}

ErrCode
LidarRenderCtxImpl::generateBboxBufs(
    const Lidar3DBbox* bbox, uint32_t num, std::vector<vec3f>& bboxBuf,
    std::vector<vec4f>& colorBuf)
{
    bboxBuf.resize(num * BBOX_3D_VERTEX_NUM);
    colorBuf.resize(num * BBOX_3D_VERTEX_NUM);
    uint32_t idx = 0;
    float cosYaw = 1.0f, sinYaw = 0.0f;
    vec3f center = {{0.0f, 0.0f}};
    auto bboxWrite = [&idx, &bboxBuf, &colorBuf, &center, &cosYaw, &sinYaw](
                         float x, float y, float z, const vec4f& color) {
        bboxBuf[idx] = {{
            (x - center.x) * cosYaw + (y - center.y) * (-sinYaw) + center.x,
            (x - center.x) * sinYaw + (y - center.y) * cosYaw + center.y,
            z,
        }};
        colorBuf[idx] = color;
        ++idx;
    };
    for (uint32_t i = 0; i < num; ++i) {
        center = {{bbox[i].centerX, bbox[i].centerY, bbox[i].centerZ}};
        float x = center.x - bbox[i].dx / 2.0f;
        float y = center.y - bbox[i].dy / 2.0f;
        float z = center.z - bbox[i].dz / 2.0f;
        float w = bbox[i].dx;
        float h = bbox[i].dy;
        float l = bbox[i].dz;
        vec4f color = bbox[i].bboxColor;

        cosYaw = cos(bbox[i].yaw);
        sinYaw = sin(bbox[i].yaw);

        bboxWrite(x, y, z, color);
        bboxWrite(x + w, y, z, color);

        bboxWrite(x + w, y, z, color);
        bboxWrite(x + w, y + h, z, color);

        // Line 3 front _
        bboxWrite(x + w, y + h, z, color);
        bboxWrite(x, y + h, z, color);

        // Line 4 front |
        bboxWrite(x, y + h, z, color);
        bboxWrite(x, y, z, color);

        // Line 1 back -
        bboxWrite(x, y, z + l, color);
        bboxWrite(x + w, y, z + l, color);

        // Line 2 back |
        bboxWrite(x + w, y, z + l, color);
        bboxWrite(x + w, y + h, z + l, color);

        // Line 3 back _
        bboxWrite(x + w, y + h, z + l, color);
        bboxWrite(x, y + h, z + l, color);

        // Line 4 back |
        bboxWrite(x, y + h, z + l, color);
        bboxWrite(x, y, z + l, color);

        // Line 1 top left /
        bboxWrite(x, y, z, color);
        bboxWrite(x, y, z + l, color);

        // Line 2 top right /
        bboxWrite(x + w, y, z + l, color);
        bboxWrite(x + w, y, z, color);

        // Line 3 bottom left /
        bboxWrite(x, y + h, z, color);
        bboxWrite(x, y + h, z + l, color);

        // Line 4 bottom right /
        bboxWrite(x + w, y + h, z, color);
        bboxWrite(x + w, y + h, z + l, color);
    }
    DS_ASSERT(idx == num * BBOX_3D_VERTEX_NUM);
    return ErrCode::kGood;
}

mat4f
LidarRenderCtxImpl::getMvp()
{
    if (!_confParsed) {
        return mvpMat(_modelMat, _viewMat, _projMat);
    }
    auto m = dsMat4ToGlm(_modelMat);
    auto v = dsMat4ToGlm(_viewMat);
    auto p = dsMat4ToGlm(_projMat);
    if (!_conf.extrinsicsMatKey.empty()) {
        m = m * dsMat4ToGlm(_frameExtrinsicMat);
    }

    glm::mat4 mvp;
    if (_conf.enableLidarTo2DImageProject) {
        auto intrinsic = dsMat4ToGlm(_frameIntrisicMat);
        glm::mat4 proj = {
            // column major
            {2.0f / _conf.imageWidth, 0.0f, 0.0f, 0.0f},
            {0.0f, -2.0f / _conf.imageHeight, 0.0f, 0.0f},
            {-1.0f, 1.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, 0.0f, 0.0f},
        };
        mvp = proj * intrinsic * m;
    } else {
        mvp = p * v * m;
    }
    return glmMat4ToDS(mvp);
}

ErrCode
LidarRenderCtxImpl::draw(GuardDataMap datamap)
{
    int32_t x = 0, y = 0;
    uint32_t w = 0, h = 0;
    uint32_t winW = 0, winH = 0;
    _window->getFrameBufferSize(winW, winH);
    w = winW;
    h = winH;
    if (_confParsed) {
        BaseRenderPriv::calculateViewPort(_normalizedView, winW, winH, x, y, w, h);
    } else {  // TODO: remove later migrate all to ensemble mode
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    size_t numPoints = 0;
    FrameGuard lidarFrame;
    if (datamap.hasData(_lidarDataName)) {
        DS3D_ERROR_RETURN(
            datamap.getGuardData(_lidarDataName, lidarFrame),
            "No pointxyz found in datamap during point cloud drawing");
        Shape pShape = lidarFrame->shape();  // N x 3(PointXYZ) || N x 4(LidarXYZI)
        DS_ASSERT(pShape.numDims);
        uint32_t frameEleSize = pShape.d[pShape.numDims - 1];
        DS_ASSERT(frameEleSize == elementSize());
        numPoints = ShapeSize(pShape) / frameEleSize;
        numPoints = std::min<size_t>(numPoints, _maxPoints);
    }

    glViewport(x, y, w, h);
    glScissor(x, y, w, h);
    glEnable(GL_SCISSOR_TEST);
    glEnable(GL_DEPTH_TEST);
    if (_confParsed && _conf.needColorClear) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    glClear(GL_DEPTH_BUFFER_BIT);

    DS_ASSERT(_pointProgram);
    DS3D_ERROR_RETURN(_pointProgram->useProgram(), "use point program failed");
    if (_conf.xRangeEnabled || _conf.yRangeEnabled || _conf.zRangeEnabled) {
        glUniform2fv(
            _pointProgram->uniformLocation("xyzRange"), 3, (const GLfloat*)&_conf.xyzRange);
    }
    mat4f mvp = getMvp();
    glUniformMatrix4fv(_pointProgram->uniformLocation("mvp"), 1, GL_FALSE, (const GLfloat*)&mvp[0]);
    glUniform4fv(_pointProgram->uniformLocation("color"), 1, (const GLfloat*)&_pointColor);
    DS3D_RETURN_GL_ERROR("set uniform data failed.");

    _pointsVao->bind();
    if (numPoints) {
        glDrawArrays(GL_POINTS, 0, numPoints);
    }
    DS3D_RETURN_GL_ERROR("draw points with color failed.");
    _pointsVao->unbind();
    _pointProgram->closeProgram();
    DS3D_RETURN_GL_ERROR("finish points program failed.");

    // draw 3d bbox
    if (datamap.hasData(_lidarBboxName) && _bboxProgram) {
        FrameGuard lidar3dBboxData;
        DS3D_ERROR_RETURN(
            datamap.getGuardData(_lidarBboxName, lidar3dBboxData),
            "No 3d bbox data: %s found in datamap during lidar data rendering",
            _lidarBboxName.c_str());

        Shape bboxShape = lidar3dBboxData->shape();
        uint32_t perBboxBytes = bboxShape.d[bboxShape.numDims - 1];
        DS_ASSERT(perBboxBytes == sizeof(Lidar3DBbox));
        uint32_t numBbox = ShapeSize(bboxShape) / perBboxBytes;
        numBbox = std::min(numBbox, _maxBboxes);

        if (numBbox) {
            glViewport(x, y, w, h);
            glScissor(x, y, w, h);
            glEnable(GL_SCISSOR_TEST);
            glClear(GL_DEPTH_BUFFER_BIT);
            glLineWidth(_conf.lineWidth);
            DS_ASSERT(_bboxProgram);
            DS3D_ERROR_RETURN(_bboxProgram->useProgram(), "use bbox program failed");
            glUniformMatrix4fv(
                _bboxProgram->uniformLocation("mvp"), 1, GL_FALSE, (const GLfloat*)&mvp[0]);
            _bboxVao->bind();
            glDrawArrays(GL_LINES, 0, numBbox * BBOX_3D_VERTEX_NUM);
            DS3D_RETURN_GL_ERROR("draw points with color failed.");
            _bboxVao->unbind();
            _bboxProgram->closeProgram();
            DS3D_RETURN_GL_ERROR("finish bbox program failed.");
        }

        if (_labelCount && _enableLabels) {
            DS3D_ERROR_RETURN(drawLabels(), "draw labels failed");
        }
    }

    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);

    return ErrCode::kGood;
}

ErrCode
LidarRenderCtxImpl::drawLabels()
{
    DS_ASSERT(_enableLabels);
    DS_ASSERT(_textProgram);
    DS3D_ERROR_RETURN(_textProgram->useProgram(), "use bbox program failed");
    auto eye = eye4x4();
    glUniformMatrix4fv(_textProgram->uniformLocation("mvp"), 1, GL_FALSE, (const GLfloat*)&eye[0]);
    int imageUniform = _textProgram->uniformLocation("imageTexture");

    glUniform1i(imageUniform, 0);
    glActiveTexture(GL_TEXTURE0);
    _textVao->bind();
    DS_ASSERT(_textImages.size() == _labelCount);
    for (uint32_t i = 0; i < _labelCount; ++i) {
        _textImages[i]->bind(GL_TEXTURE_2D);

        glDrawArrays(GL_TRIANGLES, i * LABEL_VERTEX_NUM, LABEL_VERTEX_NUM);
        DS3D_RETURN_GL_ERROR("draw label texture : %u failed.", i);
        _textImages[i]->unbind(GL_TEXTURE_2D);
    }

    _textProgram->closeProgram();
    return ErrCode::kGood;
}

ErrCode
LidarRenderCtxImpl::deinitResource()
{
    syncData();
    _cudaStream.reset();
    _textImages.clear();
    _textUvBuf.reset();
    _textVertexBuf.reset();
    _labelCount = 0;

    _bboxVao.reset();
    _pointsVao.reset();
    _bboxColorBuf.reset();
    _bboxVertBuf.reset();
    _cuPointsGlBuf.reset();
    _pointsBuf.reset();

    _textProgram.reset();
    _bboxProgram.reset();
    _pointProgram.reset();
    _confParsed = false;

    return PointCloudRenderCtxImpl::deinitResource();
}

}}  // namespace ds3d::gl

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
gl::PointCloudRender*
NvDs3D_CreateLidarRender()
{
    return new gl::LidarRenderCtxImpl();
}
DS3D_EXTERN_C_END
