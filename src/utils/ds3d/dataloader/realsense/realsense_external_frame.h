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

#ifndef _DS3D_DATALOADER_RS_EXTERNAL_FRAME__H
#define _DS3D_DATALOADER_RS_EXTERNAL_FRAME__H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wpessimizing-move"
#include <librealsense2/rs.hpp>
#pragma GCC diagnostic pop

#include "ds3d/common/hpp/frame.hpp"
#include "ds3d/common/impl/impl_frames.h"


namespace ds3d { namespace impl { namespace realsense {

Frame2DPlane inline rsFrame2Plane(rs2::video_frame& f)
{
    Frame2DPlane p{
        (uint32_t)f.get_width(), (uint32_t)f.get_height(), (uint32_t)f.get_stride_in_bytes(),
        (uint32_t)f.get_bytes_per_pixel(), 0};
    return p;
}

template <typename DataTypeTP, FrameType ft>
class RsFrameWrapper : public Frame2DBaseImpl<DataTypeTP, ft> {
public:
    template <class rsFrame>
    RsFrameWrapper(/*rs2::video_frame*/ rsFrame f)
        : Frame2DBaseImpl<DataTypeTP, ft>(
              const_cast<void*>(f.get_data()), (size_t)f.get_data_size(), {0, {0}}, MemType::kCpu,
              0, nullptr)
    {
        DS_ASSERT((size_t)f.get_bytes_per_pixel() >= sizeof(DataTypeTP));
        std::vector<Frame2DPlane> planes{rsFrame2Plane(f)};
        DS_ASSERT(bytesPerPixel<DataTypeTP>(ft) == planes[0].bytesPerPixel);
        this->setPlanes(planes);
        this->setDeleter([f = std::move(f)](void*) mutable { rs2::frame().swap(f); });
    }
};

using RSZ16DepthFrame = RsFrameWrapper<uint16_t, FrameType::kDepth>;
using RSRGBAFrame = RsFrameWrapper<uint8_t, FrameType::kColorRGBA>;

}}}  // namespace ds3d::impl::realsense

#endif  // _DS3D_DATALOADER_RS_EXTERNAL_FRAME__H