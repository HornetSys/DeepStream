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

#ifndef DS3D_VIDEO_BRIDGE_IMPL_H
#define DS3D_VIDEO_BRIDGE_IMPL_H

#include "video_bridge.h"

#include "ds3d/common/helper/memdata.h"
#include "ds3d/common/hpp/databridge.hpp"
#include "ds3d/common/impl/impl_databridge.h"

#include "video_config.h"

namespace ds3d { namespace impl { namespace bridge {

class VideoBridge2d3d : public BaseImplDataBridge {
public:
    VideoBridge2d3d() = default;
    ~VideoBridge2d3d() override;

protected:
    ErrCode processImpl(
        const struct VideoBridge2dInput* inputData, OnGuardDataCBImpl outputDataCb,
        OnGuardBridgeDataCBImpl inputConsumedCb) override;
    ErrCode stopImpl() override;
    ErrCode flushImpl() override;
    ErrCode startImpl(const std::string& content, const std::string& path);

private:
    ErrCode extractBatchMeta(NvDsBatchMeta* batchMeta, GuardDataMap& datamap,
                             ShrdPtr<VideoBridge2dInput>& inputPtr);
    void timeoutWaitProcess();

    Config _config;
    volatile bool _inProcess = false;
};

}}}  // namespace ds3d::impl::bridge


#endif  // DS3D_VIDEO_BRIDGE_IMPL_H
