/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DS3D_GLPROCESS_GL_TEXT_H
#define DS3D_GLPROCESS_GL_TEXT_H

#include "ds3d/common/common.h"
#include "ds3d/common/hpp/datamap.hpp"
#include "ds3d/common/hpp/frame.hpp"
#include "ds3d/glprocess/window.h"
#include "ds3d/glprocess/gl_buf.h"

namespace ds3d { namespace gl {

// static const char* kTextDefaultFont = "Serif Bold";
static const char* kTextDefaultFont = "Sans Bold";
static const int kTextDefaultFontSize = 12;

std::unique_ptr<Texture> CreateTextTexture(
    const std::string& text, const std::string& font = kTextDefaultFont,
    int fontSize = kTextDefaultFontSize, const vec4f& rgba = {{1, 1, 1, 1}});
}}  // namespace ds3d::gl

#endif
