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

#include "ds3d/common/func_utils.h"

#include "gl_common.h"
#include "glfw_init.h"
#include "window.h"

namespace ds3d {

namespace glfw {

struct WindowImpl {
public:
    friend class gl::GWindow;

    static void CloseCallback(GLFWwindow* w);
    static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void FrameBufferSizeChanged(GLFWwindow* window, int width, int height);
    static void MouseChanged(GLFWwindow* w, double posX, double posY);

    WindowImpl(gl::GWindow* ifc);
    ~WindowImpl() = default;

    ErrCode defaultClose();
    ErrCode defaultKeyCb(int key, int scancode, int action, int mods);
    ErrCode defaultFbSizeChanged(int width, int height);

    void destroy() { _window.reset(); }

private:
    gl::GWindow* _ifc = nullptr;
    UniqPtr<GLFWwindow> _window{nullptr, [](GLFWwindow* w) {
                                    if (w)
                                        glfwDestroyWindow(w);
                                }};
    uint32_t _width = 0;
    uint32_t _height = 0;
    std::string _title;
    std::mutex _mutex;  // lock width/height/title

    // callback functions
    GuardCB<abiWindow::CloseCB> _closeCb;
    GuardCB<abiWindow::KeyPressCB> _keyCb;
    GuardCB<abiWindow::FbSizeChangedCB> _fbSizeChanged;
    std::vector<GuardCB<abiWindow::MouseChangedCB>> _mouseChanged;
};

WindowImpl::WindowImpl(gl::GWindow* ifc) : _ifc(ifc)
{
    _closeCb.setFn<>([this]() mutable { this->defaultClose(); });
    auto keyPress = [this](int key, int scancode, int action, int mods) ->void{
        this->defaultKeyCb(key, scancode, action, mods);
    };
    _keyCb.setFn<int, int, int, int>(keyPress);
    _fbSizeChanged.setFn<int, int>(
        [this](int width, int height) { this->defaultFbSizeChanged(width, height); });
}

ErrCode
WindowImpl::defaultClose()
{
    if (_window) {
        glfwSetWindowShouldClose((GLFWwindow*)_window.get(), GLFW_TRUE);
        DS3D_RETURN_GLFW_ERROR("glfw set window should close value.");
    }
    return ErrCode::kGood;
}

ErrCode
WindowImpl::defaultKeyCb(int key, int scancode, int action, int mods)
{
    LOG_DEBUG("GLFW Key Pressed");
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        LOG_INFO("GLFW Escape key pressed");
        return this->defaultClose();
    }
    return ErrCode::kGood;
}

ErrCode
WindowImpl::defaultFbSizeChanged(int width, int height)
{
    LockMutex lock(_mutex);
    _width = width;
    _height = height;
    lock.unlock();

    glViewport(0, 0, width, height);
    DS3D_RETURN_GL_ERROR("gl viewport failed.");
    return ErrCode::kGood;
}

void
WindowImpl::CloseCallback(GLFWwindow* w)
{
    LOG_DEBUG("GLFW Close Callbacked");
    WindowImpl* impl = static_cast<WindowImpl*>(glfwGetWindowUserPointer(w));
    if (impl->_closeCb) {
        impl->_closeCb();
    } else {
        impl->defaultClose();
    }
}

void
WindowImpl::KeyCallback(GLFWwindow* w, int key, int scancode, int action, int mods)
{
    WindowImpl* impl = (WindowImpl*)glfwGetWindowUserPointer(w);
    if (impl->_keyCb) {
        impl->_keyCb(key, scancode, action, mods);
    }  else {
        impl->defaultKeyCb(key, scancode, action, mods);
    }
}

void
WindowImpl::FrameBufferSizeChanged(GLFWwindow* w, int width, int height)
{
    WindowImpl* impl = (WindowImpl*)glfwGetWindowUserPointer(w);
    if (impl->_fbSizeChanged) {
        impl->_fbSizeChanged(width, height);
    } else {
        impl->defaultFbSizeChanged(width, height);
    }
}

void
WindowImpl::MouseChanged(GLFWwindow* w, double posX, double posY)
{
    WindowImpl* impl = (WindowImpl*)glfwGetWindowUserPointer(w);
    for (auto& mc : impl->_mouseChanged) {
        if (mc) {
            mc(posX, posY);
        }
    }
}

}  // namespace glfw

namespace gl {

GWindow::GWindow()
{
    _impl = new glfw::WindowImpl(this);
}

GWindow::~GWindow()
{
    destroy();
    if (_impl) {
        delete _impl;
    }
}

void
GWindow::setSize(uint32_t w, uint32_t h)
{
    DS_ASSERT(_impl);
    LockMutex lock(_impl->_mutex);
    _impl->_width = w;
    _impl->_height = h;
}

void
GWindow::getSize(uint32_t& width, uint32_t& height) const
{
    DS_ASSERT(_impl);
    LockMutex lock(_impl->_mutex);
    width = _impl->_width;
    height = _impl->_height;
}

void
GWindow::setTitle(const char* t)
{
    DS_ASSERT(_impl);
    LockMutex lock(_impl->_mutex);
    _impl->_title = cppString(t);
}

std::string
GWindow::getTitle() const
{
    DS_ASSERT(_impl);
    LockMutex lock(_impl->_mutex);
    return _impl->_title;
}

ErrCode
GWindow::create()
{
    DS_ASSERT(_impl);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_ALPHA_BITS, 0);
    // glfwWindowHint(GLFW_STENCIL_BITS, 0);
    DS3D_RETURN_GLFW_ERROR("glfwWindow Hint failed.");

    GLFWwindow* window =
        glfwCreateWindow(_impl->_width, _impl->_height, _impl->_title.c_str(), NULL, NULL);
    DS3D_FAILED_RETURN(window, ErrCode::kGL, "glfw create window failed.");
    _impl->_window.reset(window);
    glfwSetWindowUserPointer(window, _impl);
    glfwSetWindowCloseCallback(window, glfw::WindowImpl::CloseCallback);
    glfwSetKeyCallback(window, glfw::WindowImpl::KeyCallback);
    glfwSetFramebufferSizeCallback(window, glfw::WindowImpl::FrameBufferSizeChanged);
    glfwSetCursorPosCallback(window, glfw::WindowImpl::MouseChanged);
    DS3D_RETURN_GLFW_ERROR("Create Glfw window failed.");
    DS3D_ERROR_RETURN(makeCurrent(), "make current failed");
    glfwSwapInterval(1);
    DS3D_RETURN_GLFW_ERROR("Glfw swap interval failed.");

    return ErrCode::kGood;
}

ErrCode
GWindow::makeCurrent()
{
    DS_ASSERT(_impl->_window);
    GLFWwindow* cur = glfwGetCurrentContext();
    if (cur == _impl->_window.get()) {
        return ErrCode::kGood;
    }

    glfwMakeContextCurrent(_impl->_window.get());
    DS3D_RETURN_GLFW_ERROR();
    return ErrCode::kGood;
}

void
GWindow::closeCurrent()
{
    glfwMakeContextCurrent(0);
}

ErrCode
GWindow::destroy()
{
    DS_ASSERT(_impl);
    _impl->destroy();
    return ErrCode::kGood;
}

ErrCode
GWindow::close()
{
    DS_ASSERT(_impl);
    return _impl->defaultClose();
}

void
GWindow::setCloseCallback(const abiWindow::CloseCB* closeCb)
{
    DS_ASSERT(_impl);
    _impl->_closeCb.reset(closeCb ? closeCb->refCopy(): nullptr);
}

void
GWindow::setKeyPressCallback(const abiWindow::KeyPressCB* keyCb)
{
    DS_ASSERT(_impl);
    _impl->_keyCb.reset(keyCb ? keyCb->refCopy(): nullptr);
}

void
GWindow::setFbSizeChangedCallback(const abiWindow::FbSizeChangedCB* sizeChangedCb)
{
    DS_ASSERT(_impl);
    _impl->_fbSizeChanged.reset(sizeChangedCb ? sizeChangedCb->refCopy(): nullptr);
}

void
GWindow::setMouseChangedCallback(const MouseChangedCB* mouseChangedCb)
{
    DS_ASSERT(_impl);
    _impl->_mouseChanged.clear();
    if (mouseChangedCb) {
        _impl->_mouseChanged.emplace_back(mouseChangedCb->refCopy());
    }
}

void
GWindow::appendMouseChangedCallback(const MouseChangedCB* mouseChangedCb)
{
    if (mouseChangedCb) {
        _impl->_mouseChanged.emplace_back(mouseChangedCb->refCopy());
    }
}

void*
GWindow::getNativeWindow()
{
    DS_ASSERT(_impl);
    return (void*)_impl->_window.get();
}

ErrCode
GWindow::getFrameBufferSize(uint32_t& width, uint32_t& height)
{
    DS_ASSERT(_impl->_window);
    int w = 0, h = 0;
    glfwGetFramebufferSize(_impl->_window.get(), &w, &h);
    width = (uint32_t)w;
    height = (uint32_t)h;
    DS3D_RETURN_GLFW_ERROR("glfw get framebuffer size failed");
    return ErrCode::kGood;
}

ErrCode
GWindow::getWindowSize(uint32_t& width, uint32_t& height)
{
    DS_ASSERT(_impl->_window);
    int w = 0, h = 0;
    glfwGetWindowSize(_impl->_window.get(), &w, &h);
    width = (uint32_t)w;
    height = (uint32_t)h;
    DS3D_RETURN_GLFW_ERROR("glfw get framebuffer size failed");
    return ErrCode::kGood;
}

ErrCode
GWindow::run(std::function<ErrCode()> loopRender)
{
    DS_ASSERT(_impl);
    GLFWwindow* fwWin = static_cast<GLFWwindow*>(getNativeWindow());
    DS3D_ERROR_RETURN(makeCurrent(), "make current failed in run");

    while (!glfwWindowShouldClose(fwWin)) {
        // glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        // int width = 0, height = 0;
        // glfwGetFramebufferSize(fwWin, &width, &height);
        // glViewport(0, 0, width, height);

        if (loopRender) {
            auto ret = loopRender();
            DS3D_FAILED_RETURN(isNotBad(ret), ret, "user GL render failed");
            if (!isGood(ret))
                return ret;
        }
        glfwSwapBuffers(fwWin);
        DS3D_RETURN_GLFW_ERROR("glfw SwapBuffers failed");
        LOG_DEBUG("Swapbuffer");
        glfwPollEvents();
        DS3D_RETURN_GLFW_ERROR("glfw PollEvents failed");
    }
    return ErrCode::kGood;
}

ErrCode
GWindow::runOnce(std::function<ErrCode()> rendering)
{
    DS_ASSERT(_impl);
    GLFWwindow* fwWin = static_cast<GLFWwindow*>(getNativeWindow());
    DS3D_ERROR_RETURN(makeCurrent(), "make current failed in run");
    DS3D_ERROR_RETURN(rendering(), "user GL render failed");
    glfwSwapBuffers(fwWin);
    DS3D_RETURN_GLFW_ERROR("glfw SwapBuffers failed");
    LOG_DEBUG("Swapbuffer");
    glfwPollEvents();
    DS3D_RETURN_GLFW_ERROR("glfw PollEvents failed");
    return ErrCode::kGood;
}

}  // namespace gl

}  // namespace ds3d

using namespace ds3d;

DS3D_EXTERN_C_BEGIN

abiRefGlWindow*
NvDs3D_CreateGlfwWindow(uint32_t width, uint32_t height, const char* title)
{
    DS3D_FAILED_RETURN(isGood(glfw::GlfwInstance::Init()), nullptr, "Glfw Init failed");

    std::unique_ptr<gl::GWindow> win(new gl::GWindow());
    win->setSize(width, height);
    win->setTitle(title);
    DS3D_FAILED_RETURN(isGood(win->create()), nullptr, "create glfw window failed.");
    win->makeCurrent();
    return NewAbiRef<gl::Window>(win.release());
}

DS3D_EXTERN_C_END