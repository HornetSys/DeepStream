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

#ifndef DS3D_GLPROCESS_GL_WINDOW_H
#define DS3D_GLPROCESS_GL_WINDOW_H

#include "ds3d/common/common.h"
#include "ds3d/common/hpp/obj.hpp"
#include "ds3d/common/hpp/datarender.hpp"

namespace ds3d { namespace glfw {
class WindowImpl;
}}  // namespace ds3d::glfw

namespace ds3d { namespace gl {

class Window: public abiWindow {
public:
    virtual void setSize(uint32_t width, uint32_t height) = 0;
    virtual void setTitle(const char* t) = 0;
    virtual void getSize(uint32_t& width, uint32_t& height) const = 0;
    virtual std::string getTitle() const = 0;
    virtual ErrCode makeCurrent() = 0;
    virtual void closeCurrent() = 0;
    virtual ErrCode run(std::function<ErrCode()> loopRender) = 0;
    virtual ErrCode runOnce(std::function<ErrCode()> rendering) = 0;
    virtual ErrCode close() = 0;
    virtual ErrCode getFrameBufferSize(uint32_t& width, uint32_t& height) = 0;
    virtual ErrCode getWindowSize(uint32_t& width, uint32_t& height) = 0;
};

class GWindow : public Window {
    friend class glfw::WindowImpl;

public:
    GWindow();
    virtual ~GWindow();

    void setSize(uint32_t width, uint32_t height) override;
    void setTitle(const char* t) override;
    void getSize(uint32_t& width, uint32_t& height) const override;
    std::string getTitle() const override;
    ErrCode create();
    ErrCode makeCurrent() override;
    void closeCurrent() override;
    ErrCode run(std::function<ErrCode()> loopRender) override;
    ErrCode runOnce(std::function<ErrCode()> rendering) override;
    ErrCode close() override;
    ErrCode destroy();

    ErrCode getFrameBufferSize(uint32_t& width, uint32_t& height) override;
    ErrCode getWindowSize(uint32_t& width, uint32_t& height) override;

public:
    // inherited interface
    void* getNativeWindow() override;
    void setCloseCallback(const CloseCB* closeCb) override;
    void setKeyPressCallback(const KeyPressCB* keyCb) override;
    void setFbSizeChangedCallback(const FbSizeChangedCB* sizeChangedCb) override;
    void setMouseChangedCallback(const MouseChangedCB* mouseChangedCb) override;
    void appendMouseChangedCallback(const MouseChangedCB* mouseChangedCb) override;

protected:
    glfw::WindowImpl* _impl;
};

} // namespace gl

typedef abiRefT<gl::Window> abiRefGlWindow;
typedef GuardDataT<gl::Window> GuardGlWindow;

}  // namespace ds3d

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API ds3d::abiRefGlWindow* NvDs3D_CreateGlfwWindow(
    uint32_t width, uint32_t height, const char* title);
DS3D_EXTERN_C_END


#endif  // DS3D_GLPROCESS_GL_WINDOW_H