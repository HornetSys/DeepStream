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

#ifndef DS3D_DATARENDER_GL_DATA_RENDER_IMPL_H
#define DS3D_DATARENDER_GL_DATA_RENDER_IMPL_H

#include "gles_3d_render.h"

#include "ds3d/common/helper/safe_queue.h"
#include "ds3d/common/hpp/datarender.hpp"
#include "ds3d/common/impl/impl_datarender.h"
#include "glprocess/gl_render_ctx.h"
#include "glprocess/window.h"

#include "render_config.h"

namespace ds3d { namespace impl { namespace render {

class Gl3DImplDataRender : public BaseImplDataRender {
public:
    Gl3DImplDataRender() = default;
    ~Gl3DImplDataRender() override;

protected:
    ErrCode prerollImpl(GuardDataMap datamap) override;
    ErrCode renderImpl(GuardDataMap datamap, OnGuardDataCBImpl dataDoneCb) override;
    ErrCode stopImpl() override;
    ErrCode flushImpl() override;

    ErrCode startRenderTask(const Config& config, std::unique_ptr<gl::iGLRender>&& render);

    enum class ItemType {
        kData,
        kEvent,
    };

    struct QueItem {
        ItemType type = ItemType::kData;
        std::function<ErrCode()> process;
        std::function<void()> failed;
    };

private:
    void clearQueue()
    {
        while (_renderQueue.size()) {
            QueItem item;
            try {
                item = _renderQueue.pop();
            }
            catch (const Exception& e) {
                LOG_WARNING("queue catch exception: %s", e.what());
                _renderQueue.clear();
                return;
            }
            if (item.type == ItemType::kEvent) {
                item.process();
            }
            if (item.type == ItemType::kData && item.failed) {
                item.failed();
            }
        }
    }
    virtual Config& config() = 0;

    std::unique_ptr<gl::iGLRender> _renderCtx;
    GuardGlWindow _window;
    std::unique_ptr<std::thread> _renderThread;
    SafeQueue<QueItem> _renderQueue;
    uint32_t _maxQueueSize = 4;
    bool _stopped = false;
};

class Gl3DDepthColorImplDataRender : public Gl3DImplDataRender {
public:
    Gl3DDepthColorImplDataRender() = default;
    ~Gl3DDepthColorImplDataRender() {}

protected:
    ErrCode startImpl(const std::string& content, const std::string& path) override;
    Config& config() override { return _config; }

    DepthConfig _config;
};

class Gl3DPointCloudImplDataRender : public Gl3DImplDataRender {
public:
    Gl3DPointCloudImplDataRender() = default;
    ~Gl3DPointCloudImplDataRender() {}

protected:
    ErrCode startImpl(const std::string& content, const std::string& path) override;

    Config& config() override { return _config; }
    PointsConfig _config;
};

class Gl3DLidarImplDataRender : public Gl3DImplDataRender {
public:
    Gl3DLidarImplDataRender() = default;
    ~Gl3DLidarImplDataRender() {}

protected:
    ErrCode startImpl(const std::string& content, const std::string& path) override;

    Config& config() override { return _config; }
    LidarConfig _config;
};

}}}  // namespace ds3d::impl::render


#endif  // DS3D_DATARENDER_GL_DATA_RENDER_IMPL_H