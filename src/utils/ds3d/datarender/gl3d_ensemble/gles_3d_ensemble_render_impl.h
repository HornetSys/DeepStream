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

#ifndef DS3D_DATARENDER_GLES_3D_ENSEMBLE_RENDER_IMPL_H
#define DS3D_DATARENDER_GLES_3D_ENSEMBLE_RENDER_IMPL_H

#include "gles_3d_ensemble_render.h"

#include "ds3d/common/helper/safe_queue.h"
#include "ds3d/common/helper/signalshot.h"
#include "ds3d/common/hpp/datarender.hpp"
#include "ds3d/common/impl/impl_datarender.h"
#include "ds3d/glprocess/gl_render_ctx.h"
#include "ds3d/glprocess/window.h"

#include "gles_3d_ensemble_render_config.h"

namespace ds3d { namespace impl { namespace render {

class Gl3DImplEnsembleRender : public BaseImplDataRender {
public:
    Gl3DImplEnsembleRender() = default;
    ~Gl3DImplEnsembleRender() override;

protected:
    ErrCode prerollImpl(GuardDataMap datamap) override;
    ErrCode startImpl(const std::string& content, const std::string& path) override;
    ErrCode renderImpl(GuardDataMap datamap, OnGuardDataCBImpl dataDoneCb) override;
    ErrCode stopImpl() override;
    ErrCode flushImpl() override;

    ErrCode startRenderTaskUnsafe(const EnsembleConfig& config);

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

    EnsembleConfig _config;
    std::vector<Ptr<gl::iGLRender>> _renders;
    GuardGlWindow _glfwWindow;
    std::unique_ptr<std::thread> _renderThread;
    SafeQueue<QueItem> _renderQueue;
    SignalShot _shot;
    uint32_t _maxQueueSize = 1;
    std::atomic_bool _stopped{false};
};

}}}  // namespace ds3d::impl::render


#endif  // DS3D_DATARENDER_GLES_3D_ENSEMBLE_RENDER_IMPL_H
