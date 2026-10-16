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

#include "ds3d/common/common.h"
#include "gl_common.h"

#include "gl_program.h"


namespace ds3d { namespace gl {

namespace {

std::string
programInfoLog(GLuint program)
{
    std::vector<char> strLog(4096, '\0');
    int validLen = 0;
    glGetProgramInfoLog(program, strLog.size() - 1, &validLen, &strLog[0]);
    return strLog.data();
}

}  // namespace

ErrCode
Shader::compile(const std::string& shaderStr, GLenum type)
{
    GLuint shader = glCreateShader(type);
    DS_ASSERT(shader);
    const GLchar* p = (const GLchar*)shaderStr.c_str();
    glShaderSource(shader, 1, &p, nullptr);
    glCompileShader(shader);
    // check for compile errors
    int status = -1;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (GL_TRUE != status) {
        LOG_ERROR("Compile GL Shader failed");
        std::vector<char> shaderLog(4096, '\0');
        int validLen = 0;
        glGetShaderInfoLog(shader, shaderLog.size() - 1, &validLen, &shaderLog[0]);
        LOG_ERROR("Compile Shader error log:\n %s", shaderLog.data());
        return ErrCode::kGL;
    }

    reset(shader);
    LOG_DEBUG("Shader compiled");
    return ErrCode::kGood;
}

ErrCode
Program::create()
{
    GLuint p = glCreateProgram();
    DS3D_RETURN_GL_ERROR("glCreateProgram failed");
    DS_ASSERT(p);
    reset(p);
    return ErrCode::kGood;
}

ErrCode
Program::link(const std::vector<Shader>& shaders)
{
    DS_ASSERT(get());
    for (const auto& s : shaders) {
        DS_ASSERT(s.v);
        glAttachShader(get(), s.v);
        DS3D_RETURN_GL_ERROR("glAttachShader: %u failed", s.v);
    }
    glLinkProgram(get());
    GLint status = 0;
    glGetProgramiv(get(), GL_LINK_STATUS, &status);
    if (GL_TRUE != status) {
        LOG_ERROR("Link Program error log:\n %s", programInfoLog(get()).c_str());
    }
    DS3D_FAILED_RETURN(isValid(), ErrCode::kGL, "program is invalid");
    DS3D_RETURN_GL_ERROR("gl link program: %u failed", get());
    return ErrCode::kGood;
}

bool
Program::isValid() const
{
    DS_ASSERT(get());
    glValidateProgram(get());
    GLint status = 0;
    glGetProgramiv(get(), GL_VALIDATE_STATUS, &status);
    if (GL_TRUE != status) {
        LOG_ERROR("Validate Program failed, msg:\n %s", programInfoLog(get()).c_str());
        return false;
    }
    return true;
}

Program*
BuildProgramFromSource(const std::vector<std::string>& sources, const std::vector<GLenum>& types)
{
    DS_ASSERT(sources.size() == types.size());
    DS_ASSERT(sources.size() > 0);
    std::unique_ptr<Program> p = std::make_unique<Program>();
    DS_ASSERT(p);
    DS3D_FAILED_RETURN(isGood(p->create()), nullptr, "create program failed.");
    std::vector<Shader> shaders;
    for (size_t i = 0; i < sources.size(); ++i) {
        Shader s;
        DS3D_FAILED_RETURN(
            isGood(s.compile(sources[i], types[i])), nullptr, "compile shader idx:%d failed",
            (int)i);
        shaders.emplace_back(std::move(s));
    }

    DS3D_FAILED_RETURN(
        isGood(p->link(shaders)), nullptr, "link shaders (size: %zu) failed.", shaders.size());
    DS3D_FAILED_RETURN(p->isValid(), nullptr, "Linked progam is not valid");
    return p.release();
}

}}  // namespace ds3d::gl
