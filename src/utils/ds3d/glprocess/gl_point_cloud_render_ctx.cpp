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
#include "ds3d/common/hpp/yaml_config.hpp"
#include "ds3d/glprocess/gl_helper.h"

#include "gl_point_cloud_render_ctx.h"
#include "glfw_init.h"
#include "gl_helper.h"

#include "glm/ext.hpp"
#include "glm/gtx/quaternion.hpp"

#define GLSL_VERSION "#version 300 es\n"

namespace ds3d { namespace gl {

const std::string kVertexTexScript = {
    // GLSL_VERSION
    // "#define POSITIVE_Z_ONLY \n"
    "precision mediump float;\n"
    "layout (location = 0) in vec3 inVert;\n"
    "layout (location = 2) in vec2 inTexCoord;\n"
    "uniform mat4 mvp;\n"
    "\n"
    "out vec2 TexCoord;\n"
    "\n"
    "void main()\n"
    "{\n"
    "	gl_Position = mvp * vec4(inVert, 1.0);\n"
    "#ifdef POSITIVE_Z_ONLY \n"
    "	if (inVert.z <= 0.0f) { gl_Position = vec4(2.0f, 2.0f, 2.0f, 1.0f); }\n"
    "#endif\n"

    "	TexCoord = vec2(inTexCoord.x, inTexCoord.y);\n"
    "}\n"};

const std::string kImageFragmentScript = {
    "\n"
    // GLSL_VERSION
    //"#define TEX_COORD_Y (TexCoord.y)\n"
    //"#define TEX_COORD_Y (1.0f - TexCoord.y)\n"
    "precision mediump float;\n"
    "in vec2 TexCoord;\n"
    "out vec4 FragColor;\n"
    "uniform sampler2D imageTexture;\n"
    "\n"
    "void main()\n"
    "{\n"
    "	vec2 coord = vec2(TexCoord.x, TEX_COORD_Y);\n"
    "   if(coord.x >= 0.0f && coord.x < 1.0f && coord.y >= 0.0f && coord.y < 1.0f) { \n"
    "	    FragColor = texture(imageTexture, coord);\n"
    "   } else { \n"
    "       FragColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);\n"
    "   }\n"
    "}\n"};

PointCloudRenderCtxImpl::PointCloudRenderCtxImpl()
{
    _modelMat = eye4x4();
    _viewMat = eye4x4();
    _projMat = eye4x4();
}

void
PointCloudRenderCtxImpl::setLookAt(const vec3f& position, const vec3f& target, const vec3f& up)
{
    glm::mat4 mat = glm::lookAt(vec3ToGlm(position), vec3ToGlm(target), vec3ToGlm(up));
    _viewMat = glmMat4ToDS(mat);
    _viewLookat.position = position;
    _viewLookat.target = target;
    _viewLookat.up = up;
    _useViewLookat = true;
}

void
PointCloudRenderCtxImpl::setPerspective(float degree, float ratio, float near, float far)
{
    glm::mat4 mat = glm::perspective(glm::radians(degree), ratio, near, far);
    _projMat = glmMat4ToDS(mat);
}

ErrCode
PointCloudRenderCtxImpl::configure(const char* yamlConfig, uint32_t size)
{
    PointCloudConfig conf;
    std::string yamlStr = cppString(yamlConfig, size);
    DS3D_ERROR_RETURN(BaseRenderPriv::parseConfig(yamlStr, conf), "parse base config failed.");
    YAML::Node node = YAML::Load(yamlStr);
    auto yCoordYOpp = node["coord_y_opposite"];
    auto yPositiveZonly = node["positive_z_only"];

    if (yCoordYOpp) {
        conf.coordYOpposite = yCoordYOpp.as<bool>();
    }
    if (yPositiveZonly) {
        conf.positiveZOnly = yPositiveZonly.as<bool>();
    }

    if (conf.lookatEnabled) {
        setLookAt(conf.lookat.position, conf.lookat.target, conf.lookat.up);
    }
    if (conf.perspectiveEnabled) {
        setPerspective(
            conf.perspective.fov, conf.perspective.ratio, conf.perspective.near,
            conf.perspective.far);
    }
    _conf = conf;
    _confParsed = true;
    setTexCoordY(conf.coordYOpposite);
    setPositiveZOnly(conf.positiveZOnly);
    return ErrCode::kGood;
}

ErrCode
PointCloudRenderCtxImpl::initMouseCallback(GuardGlWindow& window)
{
    DS_ASSERT(window);
    GLFWwindow* fwWin = (GLFWwindow*)window->getNativeWindow();
    // glfwSetInputMode(fwWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    DS3D_RETURN_GLFW_ERROR("Glfw disable CURSOR failed");

    GuardCB<abiWindow::MouseChangedCB> mouseCb;
    mouseCb.setFn<double, double>([this, fwWin](double x, double y) {
        static const float speedY = 0.005;
        static const float speedAngle = 0.01;
        if (glfwGetMouseButton(fwWin, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
            _lastX = x;
            _lastY = y;
            return;
        }

        float yaw = (x - _lastX) * speedAngle;
        float offsetY = (y - _lastY) * speedY;
        _lastX = x;
        _lastY = y;

        if (!_useViewLookat) {
            return;
        }

        if (yaw > 30.0f) {
            yaw = 30.0f;
        } else if (yaw < -30.0f) {
            yaw = -30.0f;
        }
        // yaw = yaw * 3.1415926f / 180.0f;

        if (_viewLookat.up.y < 0) {
            offsetY = -offsetY;
        } else {
            yaw = -yaw;
        }

        _mouseMoveYaw += yaw;

        float lookupY = _viewLookat.position.y + offsetY;

#if 1  // MODEL_ROTATION
        // rotation on target
        // applies model = rotatePoint + rotation*(P - rotatePoint)
        glm::vec3 rotateP = vec3ToGlm(_viewLookat.target);
        glm::vec3 position = vec3ToGlm(_viewLookat.position);
        position.y = lookupY;
        glm::vec3 direct = rotateP - position;
        glm::vec3 rotatePX = glm::cross(direct, vec3ToGlm(_viewLookat.up));
        // lookup Y oritention as axis
        glm::vec3 axis = glm::normalize(glm::cross(rotatePX, direct));
        glm::quat quater = angleAxis(_mouseMoveYaw, axis);
        glm::mat4 rotateMat = glm::toMat4(quater);
        glm::mat4 model = rotateMat * glm::translate(glm::mat4(1.f), -rotateP) +
                          glm::translate(glm::mat4(1.f), rotateP);
        // model = model * dsMat4ToGlm(_modelMat);
        _modelMat = glmMat4ToDS(model);
        _viewLookat.position.y = position.y;
        setLookAt(_viewLookat.position, _viewLookat.target, _viewLookat.up);
#endif

#if 0  // this one is for safe use, disable for current version
        // rotate around target
        vec3f translate{
            _viewLookat.position.x - _viewLookat.target.x, 0,
            _viewLookat.position.z - _viewLookat.target.z};
        vec3f xyz{
            (float)cos(yaw) * translate.x - (float)sin(yaw) * translate.z + _viewLookat.target.x,
            _viewLookat.position.y + offsetY,  // Y is just for offset
            (float)sin(yaw) * translate.x + (float)cos(yaw) * translate.z + _viewLookat.target.z};

        if (_useViewLookat) {
            setLookAt(xyz, _viewLookat.target, _viewLookat.up);
        }
#endif
    });
    window->appendMouseChangedCallback(mouseCb.abiRef());
    return ErrCode::kGood;
}

ErrCode
PointCloudRenderCtxImpl::initResource()
{
    if (!_window) {
        _window.reset(NvDs3D_CreateGlfwWindow(1280, 720, "point-cloud-render"));
        DS3D_FAILED_RETURN(
            _window, ErrCode::kGL, "create glfw window failed for resource creation");
    }

    DS_ASSERT(_window);
    _window->getSize(_initWindowInfo.width, _initWindowInfo.height);
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

    std::string fragmentSource;
    if (_texCoordYOpposite) {
        fragmentSource = GLSL_VERSION + std::string("#define TEX_COORD_Y (1.0f - TexCoord.y)\n") +
                         kImageFragmentScript;
    } else {
        fragmentSource =
            GLSL_VERSION + std::string("#define TEX_COORD_Y (TexCoord.y)\n") + kImageFragmentScript;
    }

    std::string vertexSource;
    if (_enablePositiveZOnly) {
        vertexSource = GLSL_VERSION + std::string("#define POSITIVE_Z_ONLY \n") + kVertexTexScript;
    } else {
        vertexSource = GLSL_VERSION + kVertexTexScript;
    }

    std::unique_ptr<Program> imageProg(BuildProgramFromSource(
        {vertexSource, fragmentSource}, {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER}));
    DS3D_FAILED_RETURN(imageProg, ErrCode::kGL, "Build Image shaders failed");
    _program = std::move(imageProg);
    DS_ASSERT(_program);

    _cudaStream.reset(new CudaStream(cudaStreamDefault, _conf.devId));
    DS_ASSERT(_cudaStream);

    DS3D_ERROR_RETURN(initTextures(), "init Textures failed.");
    DS3D_ERROR_RETURN(initVertexBuffer(), "init Vertex Buffers failed");

    return ErrCode::kGood;
}

ErrCode
PointCloudRenderCtxImpl::initVertexBuffer()
{
    // point-cloud is set by vec3f
    // coordinate data is set with vec2f

    _vertexBuf.reset(VertexBuf::create(GL_ARRAY_BUFFER));

    return ErrCode::kGood;
}

ErrCode
PointCloudRenderCtxImpl::initTextures()
{
    _colorImage.reset(Texture::create());
    DS3D_FAILED_RETURN(_colorImage, ErrCode::kGL, "Create image texture failed");
    return ErrCode::kGood;
}

ErrCode
PointCloudRenderCtxImpl::initVAO()
{
    if (!_vao) {
        _vao.reset(VertexArray::create());
        DS3D_FAILED_RETURN(_vao, ErrCode::kGL, "Create VAO failed");
    }
    DS_ASSERT(_vao);

    Program* p = _program.get();
    DS_ASSERT(p);
    DS3D_ERROR_RETURN(p->useProgram(), "use program failed");
    _inTexCoordAttrib = p->attribLocation("inTexCoord");
    _inVertAttrib = p->attribLocation("inVert");

    _vao->bind();
    _vertexBuf->bind();
    _vao->enableAttrib(_inVertAttrib);
    // stride: 0 for tightly packed
    glVertexAttribPointer(_inVertAttrib, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    _vertexBuf->unbind();
    DS3D_RETURN_GL_ERROR("enable inVert into vao failed.");

    _coordBuf->bind();
    _vao->enableAttrib(_inTexCoordAttrib);
    // stride: 0 for tightly packed
    glVertexAttribPointer(_inTexCoordAttrib, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    _coordBuf->unbind();
    _vao->unbind();
    DS3D_RETURN_GL_ERROR("enable inTexCoord into vao failed.");
    return ErrCode::kGood;
}

ErrCode
PointCloudRenderCtxImpl::uploadData(GuardDataMap datamap)
{
    Frame2DGuard imageFrame;
    DS3D_ERROR_RETURN(
        datamap.getGuardData(kColorFrame, imageFrame),
        "No image found in datamap during point cloud rendering");
    DS3D_ERROR_RETURN(uploadImage(imageFrame), "Upload color image into texture failed.");

    FrameGuard pointFrame;
    DS3D_ERROR_RETURN(
        datamap.getGuardData(kPointXYZ, pointFrame),
        "No pointxyz found in datamap during point cloud rendering");

    FrameGuard colorCoord;
    DS3D_ERROR_RETURN(
        datamap.getGuardData(kPointCoordUV, colorCoord),
        "No point color coordinate is found in datamap during point cloud rendering");
    DS3D_ERROR_RETURN(
        uploadPoints(pointFrame, colorCoord), "Upload pointcloud into vertex buffer failed.");

    return ErrCode::kGood;
}

ErrCode
PointCloudRenderCtxImpl::uploadImage(Frame2DGuard& image)
{
    DS_ASSERT(image);
    const auto& plane = image->getPlane(0);

    DS3D_FAILED_RETURN(
        image->frameType() == FrameType::kColorRGBA, ErrCode::kUnsupported,
        "pointcloud render upload image only support RGBA color");

    auto preload = [this, &plane]() {
        // glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // this is just for RGB or R
        glPixelStorei(GL_UNPACK_ROW_LENGTH, plane.pitchInBytes / plane.bytesPerPixel);
    };

    auto postload = [this]() {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    };

    DS3D_ERROR_RETURN(
        _colorImage->uploadImage(
            plane.width, plane.height, image->base(), GL_UNSIGNED_BYTE, GL_RGBA8, GL_RGBA, preload,
            postload),
        "upload depth color map failed.");

    return ErrCode::kGood;
}


ErrCode
PointCloudRenderCtxImpl::uploadPoints(FrameGuard& points, FrameGuard& colorCoord)
{
    DS_ASSERT(points);
    DS_ASSERT(colorCoord);

    DS3D_FAILED_RETURN(
        points->frameType() == FrameType::kPointXYZ && points->dataType() == DataType::kFp32,
        ErrCode::kUnsupported, "pointcloud render upload points only support PointXYZ with FP32");

    Shape pShape = points->shape();  // N x 3
    DS_ASSERT(pShape.numDims == 2 && pShape.d[1] == 3);  // PointXYZ
    size_t numPoints = (size_t)pShape.d[0];

    Shape cShape = colorCoord->shape();  // N x 3
    DS_ASSERT(cShape.numDims == 2 && cShape.d[1] == 2);  // PointColorCoord
    DS_ASSERT((int)numPoints == cShape.d[0]);
    DS3D_UNUSED(cShape);

    DS3D_FAILED_RETURN(_vertexBuf, ErrCode::kGL, "Create vertexBuffer for points failed");
    DS3D_ERROR_RETURN(
        _vertexBuf->upload<vec3f>(numPoints, (const vec3f*)points->base(), GL_STATIC_DRAW),
        "upload vertex data failed");

    _coordBuf.reset(VertexBuf::create(GL_ARRAY_BUFFER));
    DS3D_FAILED_RETURN(_coordBuf, ErrCode::kGL, "Create points coordinate buffer failed");
    DS3D_ERROR_RETURN(
        _coordBuf->upload<vec2f>(numPoints, (const vec2f*)colorCoord->base(), GL_STATIC_DRAW),
        "upload vertex data failed");

    DS3D_ERROR_RETURN(initVAO(), "init all VAO failed");

    return ErrCode::kGood;
}

ErrCode
PointCloudRenderCtxImpl::draw(GuardDataMap datamap)
{
    int32_t x = 0, y = 0;
    uint32_t w = 0, h = 0;
    Program* p = nullptr;
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

    FrameGuard pointFrame;
    DS3D_ERROR_RETURN(
        datamap.getGuardData(kPointXYZ, pointFrame),
        "No pointxyz found in datamap during point cloud drawing");
    Shape pShape = pointFrame->shape();  // N x 3
    DS_ASSERT(pShape.numDims == 2 && pShape.d[1] == 3);  // PointXYZ
    size_t numPoints = (size_t)pShape.d[0];

    p = _program.get();

    DS3D_ERROR_RETURN(p->useProgram(), "use pointcloud program failed");
    glUniform1i(p->uniformLocation("imageTexture"), 0);

    std::vector<vec4f> mvp = mvpMat(_modelMat, _viewMat, _projMat);
    glUniformMatrix4fv(p->uniformLocation("mvp"), 1, GL_FALSE, (const GLfloat*)&mvp[0]);

    glViewport(x, y, w, h);
    glScissor(x, y, w, h);
    glEnable(GL_SCISSOR_TEST);
    glEnable(GL_DEPTH_TEST);
    if (_confParsed && _conf.needColorClear) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    DS_ASSERT(p);

    glActiveTexture(GL_TEXTURE0);
    _colorImage->bind(GL_TEXTURE_2D);
    DS3D_RETURN_GL_ERROR("binding texture for point render failed.");

    // DS3D_ERROR_RETURN(p->useProgram(), "use image program failed");
    _vao->bind();
    glDrawArrays(GL_POINTS, 0, numPoints);
    DS3D_RETURN_GL_ERROR("draw points with color failed.");
    _vao->unbind();
    _colorImage->unbind(GL_TEXTURE_2D);
    // glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    p->closeProgram();

    return ErrCode::kGood;
}

ErrCode
PointCloudRenderCtxImpl::syncData()
{
    if (_cudaStream) {
        DS3D_ERROR_RETURN(_cudaStream->sync(), "PointCloudRenderCtxImpl sync data failed");
    }
    return ErrCode::kGood;
}

ErrCode
PointCloudRenderCtxImpl::deinitResource()
{
    syncData();
    _cudaStream.reset();
    _window->close();
    _vao.reset();
    _program.reset();
    _vertexBuf.reset();
    _coordBuf.reset();
    _colorImage.reset();

    return ErrCode::kGood;
}

}}  // namespace ds3d::gl

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
gl::PointCloudRender*
NvDs3D_CreatePointCloudRender()
{
    return new gl::PointCloudRenderCtxImpl();
}
DS3D_EXTERN_C_END
