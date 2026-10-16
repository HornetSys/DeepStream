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

#include "lidar_file_render_config.h"
#include "lidar_file_render_impl.h"
#include <dirent.h>

#define FILE_NAME_SIZE 1024

namespace ds3d { namespace impl { namespace lidarrender {

ErrCode
LidarDataRenderImpl::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("lidar file datarender is starting");
    ErrCode code =
        config::CatchYamlCall([&, this]() { return parseBaseConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse datarender points config: %s failed", path.c_str());
    setInputCaps(_config.compConfig.gstInCaps);
    return ErrCode::kGood;
}

ErrCode
LidarDataRenderImpl::renderImpl(GuardDataMap datamap, OnGuardDataCBImpl dataDoneCb)
{
    LOG_INFO("lidar file datarender render datamap");
    ErrCode c = ErrCode::kGood;
    TimeStamp ts{0};
    FrameGuard lidarFrame;

    dataDoneCb(ErrCode::kGood, datamap);
    if (datamap.hasData(_config.inputDatamapKey)) {
        DS3D_FAILED_RETURN(
            isGood(datamap.getGuardData(_config.inputDatamapKey, lidarFrame)), ErrCode::kNotFound,
            "get %s failed", _config.inputDatamapKey.c_str());
        DS_ASSERT(lidarFrame);
        c = datamap.getData(kTimeStamp, ts);
        if(isGood(c)) {
            LOG_DEBUG("render frame pts %lld", (long long int)ts.t0);
        }

        char fileNameString[FILE_NAME_SIZE];
        snprintf(fileNameString, FILE_NAME_SIZE, "lidar_frame_%d_%" PRIu64 ".bin", _frameNum, ts.t0);
        std::string tmpPath = _config.lidarFrameSavePath + fileNameString;
        LOG_INFO("render frame name %s bytes %d", tmpPath.c_str(),(int)lidarFrame->bytes());
        DIR *directory = opendir(_config.lidarFrameSavePath.c_str());
        if(directory == nullptr) {
            LOG_ERROR("Please check and set the right path!");
            datamap.removeData(_config.inputDatamapKey);
            return ErrCode::kNotFound;
        }
        closedir(directory);

        _dataWriter.open(tmpPath);
        DS_ASSERT(lidarFrame->base());
        if(!_dataWriter.write((const uint8_t*)lidarFrame->base(), lidarFrame->bytes())) {
            LOG_ERROR("Dump lidar data file failed!");
        }
        _dataWriter.close();

        _frameNum++;
    }

    return ErrCode::kGood;
}

ErrCode
LidarDataRenderImpl::stopImpl()
{
    LOG_INFO("lidar file datarender stopImpl");
    return ErrCode::kGood;
}

ErrCode
LidarDataRenderImpl::prerollImpl(GuardDataMap datamap)
{
    LOG_INFO("lidar file datarender prerollImpl");
    return ErrCode::kGood;
}

ErrCode
LidarDataRenderImpl::flushImpl()
{
    LOG_INFO("lidar file datarender flushImpl");
    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::lidarrender

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataRender*
createLidarFileDataRender()
{
    return NewAbiRef<abiDataRender>(new impl::lidarrender::LidarDataRenderImpl);
}
DS3D_EXTERN_C_END