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

#ifndef DS3D_DATAFILTER_SENSOR_FUSION_CLUSTER_FILTER_IMPL_H
#define DS3D_DATAFILTER_SENSOR_FUSION_CLUSTER_FILTER_IMPL_H

#include "sensorfusion_cluster_filter.h"

#include "ds3d/common/helper/safe_queue.h"
#include "ds3d/common/hpp/datafilter.hpp"
#include "ds3d/common/impl/impl_datafilter.h"
#include "ds3d/common/ds3d_analysis_datatype.h"
#include "ds3d/glprocess/gl_helper.h"

#include "sensorfusion_config.h"


namespace ds3d { namespace impl { namespace filter {

class SensorFusionClusterFilter : public BaseImplDataFilter {
public:
    SensorFusionClusterFilter() = default;
    ~SensorFusionClusterFilter() override;

protected:
    ErrCode processImpl(
        GuardDataMap datamap, OnGuardDataCBImpl outputDataCb,
        OnGuardDataCBImpl inputConsumedCb) override;
    ErrCode stopImpl() override;
    ErrCode flushImpl() override;
    ErrCode startImpl(const std::string& content, const std::string& path);

private:
    ErrCode getCalibrationData(GuardDataMap& datamap);
    ErrCode FuseDetectionCluster(
        Lidar3DBbox* obj3DBase, uint32_t num3DBbox, Object2DBbox* obj2DBase, uint32_t num2DBbox,
        std::vector<FusedDetection>& fusedVec);
    ErrCode attachFusedResults(GuardDataMap& datamap, std::vector<FusedDetection>& fusedVec);

private:
    Config _config;
    TransformMatrix _lidar2camMat = {{{{0.0f}}}};
    TransformMatrix _camIntrinsicMat = {{{{0.0f}}}};
    IntrinsicsParam _camIntrinsicParam;
    uint32_t _imageWidth = 0;
    uint32_t _imageHeight = 0;
    gl::mat4f _transform = gl::eye4x4();
    // key is 3D labels
    std::unordered_map<std::string, std::vector<MatchLabel>> _hashMatches;
};

}}}  // namespace ds3d::impl::filter


#endif  // DS3D_DATAFILTER_SENSOR_FUSION_CLUSTER_FILTER_IMPL_H
