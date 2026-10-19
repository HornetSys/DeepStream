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

#include "ds3d/glprocess/gl_text.h"

#include "ds3d/common/common.h"
#include "ds3d/common/hpp/datamap.hpp"
#include "ds3d/common/hpp/frame.hpp"
#include "ds3d/glprocess/window.h"
#include "ds3d/glprocess/gl_buf.h"
#include "ds3d/glprocess/gl_helper.h"

#include <pango/pangocairo.h>

namespace ds3d { namespace gl {
namespace {

UniqPtr<cairo_t>
CreateCairoContext(
    uint32_t width, uint32_t height, uint32_t channels, UniqPtr<cairo_surface_t>& surf,
    std::vector<uint8_t>& buffer)
{
    buffer.resize(channels * width * height);
    UniqPtr<cairo_surface_t> tmpSurf{
        cairo_image_surface_create_for_data(
            &buffer[0], CAIRO_FORMAT_ARGB32, width, height, channels * width),
        [](cairo_surface_t* s) { cairo_surface_destroy(s); }};
    UniqPtr<cairo_t> context{cairo_create(tmpSurf.get()), [](cairo_t* t) { cairo_destroy(t); }};
    surf.swap(tmpSurf);
    return context;
}

UniqPtr<cairo_t>
CreateTextContext()
{
    UniqPtr<cairo_surface_t> tmpSurf{
        cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 0, 0),
        [](cairo_surface_t* s) { cairo_surface_destroy(s); }};
    UniqPtr<cairo_t> context{cairo_create(tmpSurf.get()), [](cairo_t* t) { cairo_destroy(t); }};
    return context;
}

}  // namespace

std::unique_ptr<Texture>
CreateTextTexture(const std::string& text, const std::string& font, int fontSize, const vec4f& rgba)
{
    std::unique_ptr<Texture> txtImage;

    char fontBuf[256] = {0};
    snprintf(fontBuf, 256 - 1, "%s %d", font.c_str(), fontSize);

    auto layoutCtx = CreateTextContext();
    UniqPtr<PangoLayout> pangoLayout{
        pango_cairo_create_layout(layoutCtx.get()), [](PangoLayout* l) {
            if (l)
                g_object_unref(l);
        }};

    pango_layout_set_text(pangoLayout.get(), text.c_str(), -1);
    UniqPtr<PangoFontDescription> desc{
        pango_font_description_from_string(fontBuf), [](PangoFontDescription* d) {
            if (d)
                pango_font_description_free(d);
        }};
    pango_layout_set_font_description(pangoLayout.get(), desc.get());

    int w = 0, h = 0;
    pango_layout_get_pixel_size(pangoLayout.get(), &w, &h);
    UniqPtr<cairo_surface_t> surf;
    std::vector<uint8_t> buffer;
    auto renderCtx = CreateCairoContext(w, h, 4, surf, buffer);
    float r = rgba.x, g = rgba.y, b = rgba.z, a = rgba.w;
    cairo_set_source_rgba(renderCtx.get(), b, g, r, a);  // swap b & r
    pango_cairo_show_layout(renderCtx.get(), pangoLayout.get());

    txtImage.reset(Texture::create());
    DS3D_FAILED_RETURN(
        isGood(txtImage->uploadImage(w, h, &buffer[0], GL_UNSIGNED_BYTE, GL_RGBA8, GL_RGBA)),
        nullptr, "create GL text texture failed");

    return txtImage;
}


}}  // namespace ds3d::gl
