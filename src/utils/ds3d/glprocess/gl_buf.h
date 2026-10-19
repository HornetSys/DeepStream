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

#ifndef DS3D_GL_PROCESS_GL_BUFS_H
#define DS3D_GL_PROCESS_GL_BUFS_H

#include "ds3d/common/common.h"
#include "ds3d/glprocess/gl_common.h"

namespace ds3d { namespace gl {

class VertexBuf {
    GLuint _val = 0;
    GLenum _bufType = GL_ARRAY_BUFFER;
    size_t _maxBytes = 0;
    DS3D_DISABLE_CLASS_COPY(VertexBuf);
    explicit VertexBuf(GLenum type) : _bufType(type) {}

public:
    // type: GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER
    static VertexBuf* create(GLenum type = GL_ARRAY_BUFFER);
    ~VertexBuf() { reset(); }
    VertexBuf(VertexBuf&& o);

    void bind()
    {
        DS_ASSERT(get());
        glBindBuffer(_bufType, get());
    }
    void unbind() { glBindBuffer(_bufType, 0); }

    // usage: GL_STATIC_DRAW, GL_DYNAMIC_DRAW
    template <typename DT>
    ErrCode upload(size_t size, const DT* data, GLenum usage = GL_STATIC_DRAW)
    {
        DS_ASSERT(get());
        bind();
        glBufferData(_bufType, size * sizeof(DT), data, usage);
        _maxBytes = size * sizeof(DT);
        unbind();
        DS3D_RETURN_GL_ERROR("upload buffer failed");
        return ErrCode::kGood;
    }

    template <typename DT>
    ErrCode allocate(size_t num, GLenum usage = GL_STATIC_DRAW)
    {
        return upload<DT>(num, nullptr, usage);
    }

    ErrCode uploadSubData(int64_t offset, size_t bytes, const void* data)
    {
        DS_ASSERT(get());
        DS_ASSERT(offset + bytes <= _maxBytes);
        bind();
        glBufferSubData(_bufType, offset, bytes, data);
        unbind();
        DS3D_RETURN_GL_ERROR("upload buffer failed");
        return ErrCode::kGood;
    }

    void attribPtr(
        GLuint location, int compSize, size_t stride, GLenum type = GL_FLOAT,
        GLboolean normalized = GL_FALSE, void* offset = 0)
    {
        DS_ASSERT(get());
        bind();
        glEnableVertexAttribArray(location);
        glVertexAttribPointer(location, compSize, GL_FLOAT, normalized, stride, offset);
        // glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE,
        //                    sizeof(vertices[0]), (void*) (sizeof(float) * 2));
        unbind();
    }

    GLuint get() const { return _val; }
    void reset();
};

class VertexArray {
    GLuint _val = 0;
    DS3D_DISABLE_CLASS_COPY(VertexArray);
    VertexArray() {}

public:
    static VertexArray* create();
    ~VertexArray() { reset(); }
    VertexArray(VertexArray&& o);

    void bind()
    {
        DS_ASSERT(get());
        glBindVertexArray(get());
    }
    void unbind() { glBindVertexArray(0); }
    void enableAttrib(GLuint index) { glEnableVertexAttribArray(index); }

    GLuint get() const { return _val; }
    void reset();
};

class Program;

class Texture {
    GLuint _val = 0;
    uint32_t _width = 0;
    uint32_t _height = 0;
    DS3D_DISABLE_CLASS_COPY(Texture);
    Texture() {}

public:
    static Texture* create();
    ~Texture() { reset(); }
    Texture(Texture&& o);

    uint32_t w() const { return _width; }
    uint32_t h() const { return _height; }

    void bind(GLenum target = GL_TEXTURE_2D)
    {
        DS_ASSERT(_val);
        glBindTexture(target, _val);
    }
    void unbind(GLenum target = GL_TEXTURE_2D) { glBindTexture(target, 0); }

    ErrCode uploadImage(
        int width, int height, void* data, GLenum type = GL_UNSIGNED_BYTE,
        int internalFormat = GL_RGBA, GLenum format = GL_RGBA,
        std::function<void()> preParam = nullptr, std::function<void()> postParam = nullptr);

    ErrCode allocateImage(
        int width, int height, GLenum type, int internalFormat, int format,
        std::function<void()> preParam = nullptr, std::function<void()> postParam = nullptr)
    {
        return uploadImage(
            width, height, nullptr, type, internalFormat, format, std::move(preParam),
            std::move(postParam));
    }

    // ErrCode render(Program& p);


    GLuint get() const { return _val; }
    void reset();
};

}}  // namespace ds3d::gl

#endif  // DS3D_GL_PROCESS_GL_BUFS_H