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

#ifndef DS3D_GLPROCESS_GL_COMMON_H
#define DS3D_GLPROCESS_GL_COMMON_H

#include "ds3d/common/common.h"
#include "ds3d/common/hpp/obj.hpp"

#include <GLES3/gl3.h>
#include <EGL/egl.h>

// Include GLEW before GLFW to use the latest GL libraries:
//#define GLEW_STATIC
//#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#undef DS3D_RETURN_GLFW_ERROR
#define DS3D_RETURN_GLFW_ERROR(fmt, ...)                                                       \
    do {                                                                                       \
        const char* __description = nullptr;                                                   \
        int __glfw_code = glfwGetError(&__description);                                        \
        if (!__description)                                                                    \
            __description = "";                                                                \
        DS3D_FAILED_RETURN(                                                                    \
            __glfw_code == GLFW_NO_ERROR, ErrCode::kGL, "GLFW error: %s, " fmt, __description, \
            ##__VA_ARGS__);                                                                    \
    } while (0)


#undef DS3D_GL_FAILED_RETURN
#define DS3D_GL_FAILED_RETURN(ret, fmt, ...)                     \
    do {                                                         \
        GLenum __gl_err = glGetError();                          \
        DS3D_FAILED_RETURN(                                      \
            __gl_err == GL_NO_ERROR, ret, "GL error: %s, " fmt,  \
            ::ds3d::gl::GLErrorString(__gl_err), ##__VA_ARGS__); \
    } while (0)

#undef DS3D_RETURN_GL_ERROR
#define DS3D_RETURN_GL_ERROR(fmt, ...) DS3D_GL_FAILED_RETURN(ErrCode::kGL, fmt, ##__VA_ARGS__)

namespace ds3d { namespace gl {

using mat4f = std::vector<vec4f>;

inline const char*
GLErrorString(GLenum err)
{
    switch (err) {
#define _GL_ERROR_CASE_STR(value) \
    case value:                   \
        return #value
        _GL_ERROR_CASE_STR(GL_NO_ERROR);
        _GL_ERROR_CASE_STR(GL_INVALID_ENUM);
        _GL_ERROR_CASE_STR(GL_INVALID_VALUE);
        _GL_ERROR_CASE_STR(GL_INVALID_OPERATION);
        _GL_ERROR_CASE_STR(GL_INVALID_FRAMEBUFFER_OPERATION);
        _GL_ERROR_CASE_STR(GL_OUT_OF_MEMORY);
#undef _GL_ERROR_CASE_STR
    default:
        return "Unknown GL error";
    }
}

}}  // namespace ds3d::gl

#endif  // DS3D_GLPROCESS_GL_COMMON_H