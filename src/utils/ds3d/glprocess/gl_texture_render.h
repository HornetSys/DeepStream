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

#ifndef DS3D_GLPROCESS_GL_TEXTURE_2D_RENDER_H
#define DS3D_GLPROCESS_GL_TEXTURE_2D_RENDER_H

#include "ds3d/common/common.h"
#include "ds3d/common/hpp/datamap.hpp"
#include "ds3d/glprocess/gl_program.h"
#include "ds3d/glprocess/gl_buf.h"
#include "ds3d/glprocess/gl_render_ctx.h"
#include "ds3d/glprocess/gl_cuda_utils.h"
#include "ds3d/glprocess/gl_helper.h"

#include "gl_render_base_priv.h"

namespace ds3d { namespace gl {

static constexpr size_t kTex2DVertDefaultNum = 6;

struct Texture3DConfig : public BaseRenderConfig {
    uint32_t maxVertexNum = kTex2DVertDefaultNum;
    std::string textureFrameKey = kColorFrame;
    std::string textureVertexKey = kTextureVertexKey;
    std::string textureCoordKey = kTextureCoordinateKey;
};

class Texture3DRenderCtx : public BaseRenderPriv {
    Texture3DConfig _config;
    std::unique_ptr<VertexBuf> _vertexBuf;
    std::unique_ptr<VertexBuf> _coordBuf;

    // data allocated on 1st coming dataset
    Ptr<Texture> _texture;
    Frame2DPlane _texPlanes;
    std::unique_ptr<CudaTexture> _cudaTexture;

    // data managed during runtime
    uint32_t _vertexInUse = 0;

public:
    ErrCode configure(const char* yamlConfig, uint32_t size) override;
    ErrCode initResource() override;
    ErrCode uploadData(GuardDataMap datamap) override;
    ErrCode draw(GuardDataMap datamap) override;
    ErrCode flush() override;
    ErrCode deinitResource() override;

protected:
    Texture3DConfig& textrueConfig() { return _config; }

    ErrCode allocateBuffers();
    ErrCode allocateTexture(const Frame2DPlane& plane, MemType memType, FrameType frameType);
    ErrCode uploadDefaultValues();
    ErrCode uploadImage(GuardDataMap& datamap);
    ErrCode uploadVetex(GuardDataMap& datamap);
};

}}  // namespace ds3d::gl

#endif  // DS3D_GLPROCESS_GL_TEXTURE_2D_RENDER_H