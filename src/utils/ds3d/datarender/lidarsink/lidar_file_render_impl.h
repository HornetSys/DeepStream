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

#ifndef DS3D_DATARENDER_LIDAR_DATA_RENDER_IMPL_H
#define DS3D_DATARENDER_LIDAR_DATA_RENDER_IMPL_H

#include "lidar_file_render.h"

#include "ds3d/common/hpp/profiling.hpp"
#include "ds3d/common/hpp/datarender.hpp"
#include "ds3d/common/impl/impl_datarender.h"
#include "ds3d/common/impl/impl_frames.h"

namespace ds3d { namespace impl { namespace lidarrender {

class LidarDataRenderImpl : public BaseImplDataRender {
public:
    LidarDataRenderImpl() = default;
    ~LidarDataRenderImpl() = default;

protected:
    ErrCode prerollImpl(GuardDataMap datamap) override;
    ErrCode startImpl(const std::string& content, const std::string& path) override;
    ErrCode renderImpl(GuardDataMap datamap, OnGuardDataCBImpl dataDoneCb) override;
    ErrCode flushImpl() override;
    ErrCode stopImpl() override;

private:
    profiling::FileWriter _dataWriter;
    Config _config;
    int32_t _frameNum = 0;
};

}}}  // namespace ds3d::impl::lidarrender
#endif