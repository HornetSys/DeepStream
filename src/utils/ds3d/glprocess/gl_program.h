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

#ifndef DS3D_GLPROCESS_GL_PROGRAM_H
#define DS3D_GLPROCESS_GL_PROGRAM_H

#include "ds3d/common/common.h"
#include "ds3d/glprocess/gl_common.h"

namespace ds3d { namespace gl {

struct Shader {
    GLuint v = 0;

    Shader() = default;
    ~Shader() { reset(); }
    ErrCode compile(const std::string& shaderStr, GLenum type);

    void reset(GLuint newVal = 0)
    {
        if (v) {
            glDeleteShader(v);
        }
        v = newVal;
    }
    Shader(Shader&& o)
    {
        reset(o.v);
        o.v = 0;
    }

    DS3D_DISABLE_CLASS_COPY(Shader);
};

class Program {
    GLuint _val = 0;
    DS3D_DISABLE_CLASS_COPY(Program);

public:
    Program() = default;
    ~Program() { reset(); }
    Program(Program&& o)
    {
        reset();
        _val = o._val;
        o._val = 0;
    }
    ErrCode create();
    ErrCode link(const std::vector<Shader>& shaders);
    bool isValid() const;

    int uniformLocation(const std::string& name)
    {
        DS_ASSERT(get());
        return glGetUniformLocation(get(), name.c_str());
    }
    int attribLocation(const std::string& name)
    {
        DS_ASSERT(get());
        return glGetAttribLocation(get(), name.c_str());
    }
    ErrCode useProgram()
    {
        DS_ASSERT(get());
        glUseProgram(get());
        DS3D_RETURN_GL_ERROR("glUseProgram");
        return ErrCode::kGood;
    }

    ErrCode closeProgram()
    {
        glUseProgram(0);
        DS3D_RETURN_GL_ERROR("gl closeProgram");
        return ErrCode::kGood;
    }

    GLuint get() const { return _val; }
    void reset(GLuint newVal = 0)
    {
        if (_val) {
            glDeleteProgram(_val);
        }
        _val = newVal;
    }
};

Program* BuildProgramFromSource(
    const std::vector<std::string>& sources, const std::vector<GLenum>& types);

}}  // namespace ds3d::gl

#endif  // DS3D_GLPROCESS_GL_PROGRAM_H