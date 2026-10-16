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

#include "gl_buf.h"
#include "gl_program.h"

namespace ds3d { namespace gl {

VertexBuf::VertexBuf(VertexBuf&& o)
{
    reset();
    _val = o._val;
    _bufType = o._bufType;
    o._val = 0;
}

void
VertexBuf::reset()
{
    if (_val) {
        glDeleteBuffers(1, &_val);
    }
    _val = 0;
}

VertexBuf*
VertexBuf::create(GLenum type)
{
    GLuint id = 0;
    glGenBuffers(1, &id);
    DS3D_GL_FAILED_RETURN(nullptr, "glGenBuffers failed");
    DS_ASSERT(id);
    std::unique_ptr<VertexBuf> buf(new VertexBuf(type));
    DS_ASSERT(buf);
    buf->_val = id;
    return buf.release();
}

VertexArray::VertexArray(VertexArray&& o)
{
    reset();
    _val = o._val;
    o._val = 0;
}

void
VertexArray::reset()
{
    if (_val) {
        glDeleteVertexArrays(1, &_val);
    }
    _val = 0;
}

VertexArray*
VertexArray::create()
{
    GLuint id = 0;
    glGenVertexArrays(1, &id);
    DS3D_GL_FAILED_RETURN(nullptr, "glGenVertexArrays failed");
    DS_ASSERT(id);
    std::unique_ptr<VertexArray> array(new VertexArray());
    DS_ASSERT(array);
    array->_val = id;
    return array.release();
}

Texture*
Texture::create()
{
    GLuint id = 0;
    glGenTextures(1, &id);
    DS3D_GL_FAILED_RETURN(nullptr, "glGenTextures failed");
    DS_ASSERT(id);
    std::unique_ptr<Texture> tex(new Texture());
    DS_ASSERT(tex);
    tex->_val = id;
    return tex.release();
}

Texture::Texture(Texture&& o)
{
    reset();
    _val = o._val;
    o._val = 0;
}

void
Texture::reset()
{
    if (_val) {
        glDeleteTextures(1, &_val);
    }
    _val = 0;
}

ErrCode
Texture::uploadImage(
    int w, int h, void* data, GLenum type, int internalFormat, GLenum format,
    std::function<void()> preParam, std::function<void()> postParam)
{
    DS_ASSERT(get());
    bind(GL_TEXTURE_2D);
    if (preParam) {
        preParam();
        DS3D_RETURN_GL_ERROR("uploadImage preParam");
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    DS3D_RETURN_GL_ERROR("uploadImage preParam");

    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, data);
    DS3D_RETURN_GL_ERROR("glTexImage2D");
    // Z16
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, width, height, 0, GL_RG, GL_UNSIGNED_BYTE, data)
    // RGBA
    // glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    if (postParam) {
        postParam();
        DS3D_RETURN_GL_ERROR("uploadImage failed in postParam");
    }
    _width = (uint32_t)w;
    _height = (uint32_t)h;

    unbind(GL_TEXTURE_2D);

    DS3D_RETURN_GL_ERROR("uploadImage");
    return ErrCode::kGood;
}

#if 0
ErrCode Texture::render(Program& p) {
    DS_ASSERT(get());
    GLuint tex = get();
    DS3D_ERROR_RETURN(p.useProgram(), "use program");
    int m = p.uniformLocation("matrix");
    DS_ASSERT(m > 0);
    int v = p.attribLocation("vao");
    DS_ASSERT(v > 0);

    // VBO, VAO

    // activate texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindTexture(GL_TEXTURE_2D, 0);
    DS3D_RETURN_GL_ERROR("active texture");

    glUseProgram(0);
    return ErrCode::kGood;
}
#endif


}}  // namespace ds3d::gl
