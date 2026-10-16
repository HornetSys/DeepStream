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

#include "gl_common.h"
#include "glfw_init.h"

namespace ds3d { namespace glfw {


static void
gErrorCallback(int error, const char* description)
{
    LOG_ERROR("GLFW Error: %d detected during glfw runtime, description: %s\n", error, description);
}

std::mutex GlfwInstance::_gMutex;
bool GlfwInstance::gInitialized = false;

ErrCode
GlfwInstance::Init()
{
    LockMutex lock(_gMutex);
    if (gInitialized) {
        return ErrCode::kGood;
    }
    glfwSetErrorCallback(gErrorCallback);
    glfwInit();
    DS3D_RETURN_GLFW_ERROR("glfwInit failed");
    gInitialized = true;
    return ErrCode::kGood;
}

ErrCode
GlfwInstance::Deinit()
{
    LockMutex lock(_gMutex);
    if (!gInitialized) {
        return ErrCode::kGood;
    }
    glfwTerminate();
    gInitialized = false;
    DS3D_RETURN_GLFW_ERROR("glfw Terminated");
    return ErrCode::kGood;
}
}}  // namespace ds3d::glfw
