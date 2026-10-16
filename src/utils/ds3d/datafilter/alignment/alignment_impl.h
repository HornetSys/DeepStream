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

#ifndef DS3D_DATAFILTER_ALIGNMENT_IMPL_H
#define DS3D_DATAFILTER_ALIGNMENT_IMPL_H

#include "alignment.h"

#include "ds3d/common/helper/memdata.h"
#include "ds3d/common/hpp/datafilter.hpp"
#include "ds3d/common/impl/impl_datafilter.h"
#include "ds3d/glprocess/gl_render_ctx.h"
#include "ds3d/glprocess/window.h"

#include "alignment_config.h"

namespace ds3d { namespace impl { namespace filter {

using MemPtr = std::unique_ptr<MemData>;

class AlignmentFilter : public BaseImplDataFilter {
public:
    AlignmentFilter() = default;
    ~AlignmentFilter() override;

protected:
    ErrCode processImpl(
        GuardDataMap datamap, OnGuardDataCBImpl outputDataCb, OnGuardDataCBImpl inputConsumedCb) override;
    ErrCode stopImpl() override;
    ErrCode flushImpl() override;
    ErrCode startImpl(const std::string& content, const std::string& path);

    ErrCode startTask(const Config& config, std::unique_ptr<gl::iGLRender>&& filter);

private:
    ErrCode addConfigDataTo(GuardDataMap& datamap);
    ErrCode copyFromRetentionTo(GuardDataMap& datamap);
    Config _config;
    IntrinsicsParam _camIntrinsics;
    // pool for aligned lidar data (output) buffers
    std::shared_ptr<BufferPool<MemPtr>> _alignedOpPool;
    // retain all unique keys ever output from alignment filter and its latest value
    GuardDataMap retentionDataMap;
};

}}}  // namespace ds3d::impl::filter


#endif  // DS3D_DATAFILTER_ALIGNMENT_IMPL_H
