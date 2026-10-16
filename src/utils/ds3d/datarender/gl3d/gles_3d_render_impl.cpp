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

#include "gles_3d_render.h"
#include "gles_3d_render_impl.h"

#include "ds3d/common/helper/safe_queue.h"
#include "ds3d/common/hpp/datarender.hpp"
#include "ds3d/common/impl/impl_datarender.h"
#include "glprocess/gl_render_ctx.h"
#include "glprocess/window.h"

#include "render_config.h"
#include <unistd.h>

namespace ds3d { namespace impl { namespace render {

Gl3DImplDataRender::~Gl3DImplDataRender()
{
    stop_i();
}

ErrCode
Gl3DImplDataRender::startRenderTask(
    const Config& config, std::unique_ptr<gl::iGLRender>&& renderCtx)
{
    LOG_INFO("gl3d datarender is starting");
    std::unique_lock<ImplMutex> locker(mutex());

    DS_ASSERT(!_window);
    _window.reset(NvDs3D_CreateGlfwWindow(config.width, config.height, config.title.c_str()));
    DS3D_FAILED_RETURN(_window, ErrCode::kGL, "create glfw window failed during start");
    DS_ASSERT(renderCtx);
    renderCtx->setWindow(_window);
    DS3D_ERROR_RETURN(renderCtx->initResource(), "init render context resource failed");

    _renderCtx = std::move(renderCtx);
    setInputCaps(config.compConfig.gstInCaps);
    setWindow(_window.cast<GuardWindow>());
    LOG_DEBUG("gl3d datarender is started");

    _window->closeCurrent();
    _stopped = false;

    auto timeoutRender = [this]() {
        try {
            auto item = _renderQueue.pop(30);
            if (item.process) {
                return item.process();
            }
        }
        catch (const Exception& e) {
            if (e.code() == ErrCode::kTimeOut) {
                return ErrCode::kGood;
            }
            if (e.code() == ErrCode::kLockWakeup) {
                return ErrCode::KEndOfStream;
            }
            return e.code();
        }
        return ErrCode::kGood;
    };

    // render thread to run window rendering
    _renderThread = std::make_unique<std::thread>([this, timeoutRender]() {
        ErrCode c = _window->run(timeoutRender);
        if (!isNotBad(c)) {
            LOG_ERROR("glwinow run failed.");
            emitError(c, "GLwindow run failed");
        } else {
            LOG_INFO("EndOfStream Window closed");
        }
        std::unique_lock<ImplMutex> locker(mutex());
        _stopped = true;
        locker.unlock();
        this->clearQueue();
    });

    return ErrCode::kGood;
}

ErrCode
Gl3DImplDataRender::prerollImpl(GuardDataMap datamap)
{
    LOG_DEBUG("gl3d datarender prerolled");
    std::unique_lock<ImplMutex> locker(mutex());
    return ErrCode::kGood;
}

ErrCode
Gl3DImplDataRender::renderImpl(GuardDataMap datamap, OnGuardDataCBImpl dataDoneCb)
{
    LOG_DEBUG("gl3d datarender is starting to render datamap");
    std::unique_lock<ImplMutex> locker(mutex());
    while (_renderQueue.size() > _maxQueueSize && !_stopped) {
        if (!config().block) {
            LOG_WARNING("dropping buffer");
            dataDoneCb(ErrCode::kByPass, datamap);
            return ErrCode::kByPass;
        }
        locker.unlock();
        usleep(20 * 1000);
        locker.lock();
    }
    if (_stopped) {
        return ErrCode::KEndOfStream;
    }
    auto renderF = [this, datamap, dataDoneCb]() {
        DS_ASSERT(_renderCtx);
        DS3D_ERROR_RETURN(_renderCtx->uploadData(datamap), "upload datamap into Gl failed.");
        DS3D_ERROR_RETURN(_renderCtx->syncData(), "Gl3DImplDataRender sync data into Gl failed.");
        dataDoneCb(ErrCode::kGood, datamap);
        DS3D_ERROR_RETURN(_renderCtx->draw(datamap), "draw data failed.");
        return ErrCode::kGood;
    };
    auto failed = [this, datamap, dataDoneCb]() {
        dataDoneCb(ErrCode::kUnknown, std::move(datamap));
    };

    QueItem item{ItemType::kData, renderF, failed};
    _renderQueue.push(std::move(item));

    return ErrCode::kGood;
}

ErrCode
Gl3DImplDataRender::stopImpl()
{
    LOG_INFO("gl3d datarender is closing");
    std::unique_lock<ImplMutex> locker(mutex());
    _stopped = true;
    if (_window) {
        _window->close();
    }
    auto thread = std::move(_renderThread);
    _renderQueue.wakeupOnce();
    locker.unlock();

    if (thread) {
        thread->join();
        thread.reset();
    }
    if (_renderCtx) {
        _renderCtx->deinitResource();
    }
    clearQueue();

    return ErrCode::kGood;
}

ErrCode
Gl3DImplDataRender::flushImpl()
{
    LOG_INFO("gl3d datarender is flushing");

    std::unique_lock<ImplMutex> locker(mutex());
    if (_renderThread && !_stopped) {
        auto p = std::make_shared<std::promise<void>>();
        std::future<void> f = p->get_future();
        auto sync = [this, p]() {
            p->set_value();
            return ErrCode::kGood;
        };
        QueItem item{ItemType::kEvent, sync, nullptr};
        _renderQueue.push(item);
        locker.unlock();
        f.wait_for(std::chrono::seconds(1));
        // enable for error debug purpose
        // f.wait();
    }
    return ErrCode::kGood;
}

ErrCode
Gl3DDepthColorImplDataRender::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("gl3d depthcolor datarender is starting");
    std::unique_lock<ImplMutex> locker(mutex());
    ErrCode code = config::CatchYamlCall(
        [&, this]() { return parseDepthColorConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse datarender depth config: %s failed", path.c_str());
    DS3D_FAILED_RETURN(
        _config.check(), ErrCode::kConfig, "check datarender depth config: %s failed",
        path.c_str());

    std::unique_ptr<gl::DepthRender> renderCtx(
        NvDs3D_CreateVideoDepthRender(_config.hasColorStream(), _config.hasDepthStream()));
    DS_ASSERT(renderCtx);
    renderCtx->setDepthRange(_config.minDepth, _config.maxDepth);
    renderCtx->setDepthColor(_config.minDepthColor, _config.maxDepthColor);

    return startRenderTask(_config, std::move(renderCtx));
}

ErrCode
Gl3DPointCloudImplDataRender::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("gl3d pointcloud datarender is starting");
    std::unique_lock<ImplMutex> locker(mutex());
    ErrCode code =
        config::CatchYamlCall([&, this]() { return parsePointsConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse datarender points config: %s failed", path.c_str());
    DS3D_FAILED_RETURN(
        _config.check(), ErrCode::kConfig, "check datarender points config: %s failed",
        path.c_str());

    std::unique_ptr<gl::PointCloudRender> renderCtx(NvDs3D_CreatePointCloudRender());
    DS_ASSERT(renderCtx);
    renderCtx->setLookAt(_config.viewPosition, _config.viewTarget, _config.viewUp);
    renderCtx->setPerspective(
        _config.fov, _config.width / (float)_config.height, _config.near, _config.far);
    renderCtx->setTexCoordY(_config.coordYOpposite);
    renderCtx->setPositiveZOnly(_config.positiveZOnly);

    return startRenderTask(_config, std::move(renderCtx));
}

ErrCode
Gl3DLidarImplDataRender::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("gl3d pointcloud datarender is starting");
    std::unique_lock<ImplMutex> locker(mutex());
    ErrCode code =
        config::CatchYamlCall([&, this]() { return parseLidarConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse datarender points config: %s failed", path.c_str());
    DS3D_FAILED_RETURN(
        _config.check(), ErrCode::kConfig, "check datarender points config: %s failed",
        path.c_str());

    std::unique_ptr<gl::PointCloudRender> renderCtx(NvDs3D_CreateLidarRender());
    DS_ASSERT(renderCtx);
    renderCtx->setLookAt(_config.viewPosition, _config.viewTarget, _config.viewUp);
    renderCtx->setPerspective(
        _config.fov, _config.width / (float)_config.height, _config.near, _config.far);
    renderCtx->setElementSize(_config.elementSize);
    renderCtx->setPointsDataKey(_config.lidarDataKey);
    renderCtx->set3DBboxKey(_config.lidarBboxKey);
    renderCtx->setLidarDataColor(_config.lidarColor);
    renderCtx->enableLabels(_config.enableLabel);

    return startRenderTask(_config, std::move(renderCtx));
}

}}}  // namespace ds3d::impl::render

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataRender*
createDepthStreamDataRender()
{
    return NewAbiRef<abiDataRender>(new impl::render::Gl3DDepthColorImplDataRender);
}

DS3D_EXPORT_API ds3d::abiRefDataRender*
createPointCloudDataRender()
{
    return NewAbiRef<abiDataRender>(new impl::render::Gl3DPointCloudImplDataRender);
}

DS3D_EXPORT_API ds3d::abiRefDataRender*
createLidarDataRender()
{
    return NewAbiRef<abiDataRender>(new impl::render::Gl3DLidarImplDataRender);
}

DS3D_EXTERN_C_END