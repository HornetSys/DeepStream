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

#ifndef DS3D_GLPROCESS_GL_HELPER_H
#define DS3D_GLPROCESS_GL_HELPER_H

#include "ds3d/common/common.h"
#include "ds3d/common/hpp/obj.hpp"

#include "ds3d/glprocess/gl_common.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace ds3d { namespace gl {

inline mat4f
eye4x4()
{
    mat4f ret;
    ret.resize(4, {{0.0f, 0.0f, 0.0f, 0.0f}});
    ret[0].data[0] = 1.0f;
    ret[1].data[1] = 1.0f;
    ret[2].data[2] = 1.0f;
    ret[3].data[3] = 1.0f;
    DS_ASSERT(ret.size() == 4);
    return ret;
}

inline mat4f
glmMat4ToDS(const glm::mat4& m)
{
    mat4f ret(4);
    memcpy(&ret[0], glm::value_ptr(m), sizeof(float) * 16);
    DS_ASSERT(ret.size() == 4);
    return ret;
}

inline glm::mat4
dsMat4ToGlm(const mat4f& m)
{
    DS_ASSERT(m.size() == 4);
    glm::mat4 ret(1.0f);
    memcpy(glm::value_ptr(ret), &m[0], sizeof(float) * 16);
    return ret;
}

inline glm::vec3
vec3ToGlm(const vec3f& m)
{
    glm::vec3 ret(0.0f);
    memcpy(glm::value_ptr(ret), &m, sizeof(vec3f));
    return ret;
}

inline mat4f
mvpMat(const mat4f& model, const mat4f& view, const mat4f& project)
{
    glm::mat4 m = glm::make_mat4<float>((const float*)&model[0]);
    glm::mat4 v = glm::make_mat4<float>((const float*)&view[0]);
    glm::mat4 p = glm::make_mat4<float>((const float*)&project[0]);
    return glmMat4ToDS(p * v * m);
}

inline mat4f
mat4Mul(const mat4f& a, const mat4f& b)
{
    DS_ASSERT(a.size() >= 4 && b.size() >= 4);
    glm::mat4 gA = glm::make_mat4<float>((const float*)&a[0]);
    glm::mat4 gB = glm::make_mat4<float>((const float*)&b[0]);
    return glmMat4ToDS(gA * gB);
}

}}  // namespace ds3d::gl

#endif  // DS3D_GLPROCESS_GL_HELPER_H