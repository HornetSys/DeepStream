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

#include "gles_3d_ensemble_render.h"
#include "gles_3d_ensemble_render_impl.h"

#include "ds3d/common/helper/safe_queue.h"
#include "ds3d/common/hpp/datarender.hpp"
#include "ds3d/common/impl/impl_datarender.h"
#include "ds3d/glprocess/gl_common.h"

#include <unistd.h>

namespace ds3d { namespace impl { namespace render {

Gl3DImplEnsembleRender::~Gl3DImplEnsembleRender()
{
    stop_i();
}

ErrCode
Gl3DImplEnsembleRender::startRenderTaskUnsafe(const EnsembleConfig& config)
{
    LOG_INFO("gles_ensemble_render is starting tasks");

    auto timeoutRender = [this]() {
        try {
            auto item = _renderQueue.pop(30);
            _shot.signal();
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
        std::unique_lock<ImplMutex> locker(mutex());
        GuardGlWindow win = _glfwWindow;
        locker.unlock();

        ErrCode c = _glfwWindow->run(timeoutRender);
        if (!isNotBad(c)) {
            LOG_ERROR("glwinow run failed.");
            emitError(c, "GLwindow run failed");
        } else {
            LOG_INFO("EndOfStream Window closed");
        }

        locker.lock();
        _stopped = true;
        locker.unlock();
        this->clearQueue();
    });

    return ErrCode::kGood;
}

ErrCode
Gl3DImplEnsembleRender::prerollImpl(GuardDataMap datamap)
{
    LOG_DEBUG("gles_ensemble_render prerolled");
    std::unique_lock<ImplMutex> locker(mutex());
    return ErrCode::kGood;
}

ErrCode
Gl3DImplEnsembleRender::renderImpl(GuardDataMap datamap, OnGuardDataCBImpl dataDoneCb)
{
    LOG_DEBUG("gles_ensemble_render is starting to render datamap");
    std::unique_lock<ImplMutex> locker(mutex());
    while (_renderQueue.size() > _maxQueueSize && !_stopped) {
        if (!_config.block) {
            LOG_WARNING("dropping buffer");
            dataDoneCb(ErrCode::kByPass, datamap);
            return ErrCode::kByPass;
        }
        locker.unlock();
        _shot.wait(100);
        locker.lock();
    }
    if (_stopped) {
        return ErrCode::KEndOfStream;
    }
    auto renderF = [this, datamap, dataDoneCb]() {
        std::unique_lock<ImplMutex> locker(mutex());
        DS_ASSERT(_renders.size());
        auto renders = _renders;
        GuardGlWindow window = _glfwWindow;
        bool clearWindow = _config.winClear;
        locker.unlock();

        if (_stopped) {
            dataDoneCb(ErrCode::kState, datamap);
            return ErrCode::kState;
        }

        LOG_DEBUG("ensemble render is drawing datamap with keys: ");
        datamap.printDebug();

        std::vector<ErrCode> upRes(renders.size(), ErrCode::kGood);
        for (size_t i = 0; i < renders.size(); ++i) {
            auto& r = renders[i];
            ErrCode c = r->uploadData(datamap);
            upRes[i] = c;
            DS3D_FAILED_RETURN(isNotBad(c), c, "upload datamap into Gl failed.");
        }
        if (clearWindow) {
            uint32_t w = 0, h = 0;
            window->getFrameBufferSize(w, h);
            glViewport(0, 0, w, h);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        // synchronize all data together to improve performance
        for (size_t i = 0; i < renders.size(); ++i) {
            auto& r = renders[i];
            if (!isGood(upRes[i])) {  // skip ErrCode >= ErrCode::KGood
                continue;
            }
            ErrCode c = r->syncData();
            DS3D_FAILED_RETURN(isNotBad(c), c, "sync data failed.");
        }
        for (size_t i = 0; i < renders.size(); ++i) {
            auto& r = renders[i];
            if (!isGood(upRes[i])) {  // skip ErrCode >= ErrCode::KGood
                continue;
            }
            ErrCode c = r->draw(datamap);
            DS3D_FAILED_RETURN(isNotBad(c), c, "draw data failed.");
        }
        dataDoneCb(ErrCode::kGood, datamap);
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
Gl3DImplEnsembleRender::stopImpl()
{
    LOG_INFO("gles_ensemble_render is closing");
    std::unique_lock<ImplMutex> locker(mutex());
    auto renders = _renders;
    auto thread = std::move(_renderThread);
    _stopped = true;
    if (_glfwWindow) {
        _glfwWindow->close();
    }
    _renders.clear();

    _renderQueue.wakeupOnce();
    locker.unlock();

    if (thread) {
        thread->join();
        thread.reset();
    }
    for (auto& r : renders) {
        r->deinitResource();
    }
    clearQueue();

    return ErrCode::kGood;
}

ErrCode
Gl3DImplEnsembleRender::flushImpl()
{
    LOG_INFO("gles_ensemble_render is flushing");

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
        _shot.signal();
        f.wait_for(std::chrono::seconds(1));
        // enable for error debug purpose
        // f.wait();
    }
    return ErrCode::kGood;
}

ErrCode
Gl3DImplEnsembleRender::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("gles_ensemble_render is starting");
    EnsembleConfig config;
    ErrCode code =
        config::CatchYamlCall([&, this]() { return parseConfig(content, path, config); });
    DS3D_ERROR_RETURN(code, "parse gles_ensemble_render config: %s failed", path.c_str());
    DS3D_FAILED_RETURN(
        config.ensembleContents.size() > 0, ErrCode::kConfig,
        "No render_graph parsed from config: %s failed", path.c_str());

    GuardGlWindow window;
    window.reset(
        NvDs3D_CreateGlfwWindow(config.winWidth, config.winHeight, config.winTitle.c_str()));
    DS3D_FAILED_RETURN(
        window, ErrCode::kGL, "create glfw window: %s failed during start",
        config.winTitle.c_str());

    std::vector<Ptr<gl::iGLRender>> renders;
    for (const auto& conf : config.ensembleContents) {
        const std::string& content = std::get<1>(conf);
        Ptr<gl::iGLRender> render;
        switch (std::get<0>(conf)) {
        case RenderType::kTexture3D:
            render.reset(NvDs3D_CreateTexture3DRender(content.c_str(), content.size()));
            break;
        case RenderType::kLidar3D:
            render.reset(NvDs3D_CreateLidarRender());
            break;
        case RenderType::kPointCloudImage:
            render.reset(NvDs3D_CreatePointCloudRender());
            break;
        case RenderType::kDepthColor:
            render.reset(NvDs3D_CreateVideoDepthRender(false, true));
            break;
        default:
            LOG_ERROR("failed to create render, unsupported rendertype");
            return ErrCode::kUnsupported;
        }
        DS3D_FAILED_RETURN(render, ErrCode::kNotFound, "create render context failed");
        render->setWindow(window);
        DS3D_ERROR_RETURN(
            render->configure(content.c_str(), content.size()),
            "configure render context failed with: %s", content.c_str());
        DS3D_ERROR_RETURN(render->initResource(), "init render context resource failed");
        renders.emplace_back(std::move(render));
    }
    DS_ASSERT(renders.size());

    std::unique_lock<ImplMutex> locker(mutex());
    DS3D_FAILED_RETURN(
        !_stopped, ErrCode::kUnknown,
        "gles_ensemble_render failed to start since stop happened too early");
    _renders = std::move(renders);
    _config = config;
    _glfwWindow = window;
    DS_ASSERT(_glfwWindow);
    setInputCaps(config.compConfig.gstInCaps);
    setWindow(window.cast<GuardWindow>());
    _glfwWindow->closeCurrent();
    _stopped = false;

    return startRenderTaskUnsafe(_config);
}

}}}  // namespace ds3d::impl::render

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataRender*
NvDs3D_CreateGlesEnsembleRender()
{
    return NewAbiRef<abiDataRender>(new impl::render::Gl3DImplEnsembleRender);
}

DS3D_EXTERN_C_END
