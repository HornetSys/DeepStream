/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DS3D_GLPROCESS_GL_SCRIPT_H
#define DS3D_GLPROCESS_GL_SCRIPT_H

#include "ds3d/common/hpp/frame.hpp"

#include "ds3d/glprocess/gl_program.h"
#include "ds3d/glprocess/gl_buf.h"

#define GLSL_VERSION "#version 300 es\n"

namespace ds3d { namespace gl {

static const std::string kImageTextureVertexScript = {
    // GLSL_VERSION
    "precision mediump float;\n"
    "layout (location = 0) in vec3 inVert;\n"
    "layout (location = 1) in vec2 inTexCoord;\n"
    "uniform mat4 mvp;\n"
    "\n"
    "out vec2 TexCoord;\n"
    "\n"
    "void main()\n"
    "{\n"
    "	gl_Position = mvp * vec4(inVert, 1.0);\n"
    "	TexCoord = vec2(inTexCoord.x, inTexCoord.y);\n"
    "}\n"};

static const std::string kImageTextureFragmentScript = {
    // GLSL_VERSION
    "precision mediump float;\n"
    "in vec2 TexCoord;\n"
    "out vec4 FragColor;\n"
    "uniform sampler2D imageTexture;\n"
    "\n"
    "void main()\n"
    "{\n"
    "	vec2 finalCoord = vec2(TexCoord.x, 1.0 - TexCoord.y);\n"
    "	FragColor = texture(imageTexture, finalCoord);\n"
    "}\n"};


static const std::string kLidarVertexScript = {
    // GLSL_VERSION
    // "#define PointType vec4 \n"
    "precision mediump float;\n"
    "layout (location = 0) in PointType point;\n"
    "uniform mat4 mvp;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    gl_Position = mvp * vec4(point.x, point.y, point.z, 1.0);\n"
    "}\n"};

const std::string kLidarZRangeVertexScript = {
    // GLSL_VERSION
    // "#define PointType vec4 \n"
    // "#define Z_RANGE \n"
    // "#define Y_RANGE \n"
    // "#define X_RANGE \n"
    "precision mediump float;\n"
    "layout (location = 0) in PointType point;\n"
    "uniform mat4 mvp;\n"
    "uniform vec2 xyzRange[3];\n"
    "\n"
    "void main()\n"
    "{\n"
    "    gl_Position = mvp * vec4(point.xyz, 1.0f);\n"
    "#ifdef Z_RANGE \n"
    "    if (point.z < xyzRange[2].x || point.z > xyzRange[2].y) { \n"
    "       gl_Position = vec4(2.0f, 2.0f, 2.0f, 1.0f);\n"
    "   }\n"
    "#endif \n"
    "#ifdef Y_RANGE \n"
    "    if (point.y < xyzRange[1].x || point.y > xyzRange[1].y) { \n"
    "       gl_Position = vec4(2.0f, 2.0f, 2.0f, 1.0f);\n"
    "   }\n"
    "#endif \n"
    "#ifdef X_RANGE \n"
    "    if (point.x < xyzRange[0].x || point.x > xyzRange[0].y) { \n"
    "       gl_Position = vec4(2.0f, 2.0f, 2.0f, 1.0f);\n"
    "   }\n"
    "#endif \n"
    "}\n"};

static const std::string kLidarFragmentScript = {
    "\n"
    // GLSL_VERSION
    "precision mediump float;\n"
    "out vec4 FragColor;\n"
    "uniform vec4 color;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    FragColor = color;\n"
    "}\n"};


static const std::string kLidarBboxVertexScript = {
    // GLSL_VERSION
    "precision mediump float;\n"
    "layout (location = 0) in vec3 vert;\n"
    "layout (location = 1) in vec4 color;\n"
    "uniform mat4 mvp;\n"
    "out vec4 fColor;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    gl_Position = mvp * vec4(vert.x, vert.y, vert.z, 1.0);\n"
    "    fColor = color;\n"
    "}\n"};

static const std::string kLidarBboxFragmentScript = {
    "\n"
    // GLSL_VERSION
    "precision mediump float;\n"
    "in vec4 fColor;\n"
    "out vec4 FragColor;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    FragColor = fColor;\n"
    "}\n"};


}}  // namespace ds3d::gl

#endif