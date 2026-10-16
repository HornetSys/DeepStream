#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert a GFM-style DeepStream report to a styled HTML file and then
to PDF via wkhtmltopdf.

Images referenced as ![alt](file.png) are resolved relative to the markdown
file's directory and embedded as base64 data URIs so the HTML is self-contained.

Usage:
    python3 md-to-html-pdf.py <report.md> <style.css> <output_dir> [model_name]
        [--report-kind benchmark|validation|status]

    model_name (optional): if provided, PDF is named benchmark_report_{model_name}.pdf
                           if omitted, derived from output_dir parent folder name

The default report kind is benchmark to preserve existing Step 8 callers.
"""
import argparse
import html as html_module
import sys
import os
import re
import base64
import shutil
import subprocess
import markdown

def find_wkhtmltopdf():
    """Return the configured renderer without mutating system package state."""
    return shutil.which('wkhtmltopdf')

def embed_images(html: str, base_dir: str) -> str:
    """Replace <img src="file.png"> with base64-embedded data URIs."""
    def replacer(match):
        prefix = match.group(1)
        src = match.group(2)
        suffix = match.group(3)
        # Skip URLs and absolute paths
        if re.match(r'^(https?|data|ftp)://', src) or os.path.isabs(src):
            return match.group(0)
        img_path = os.path.realpath(os.path.join(base_dir, src))
        base_real = os.path.realpath(base_dir)
        # Reject path traversal outside base_dir
        if not img_path.startswith(base_real + os.sep) and img_path != base_real:
            return match.group(0)
        if os.path.isfile(img_path):
            ext = os.path.splitext(src)[1].lstrip('.').lower()
            mime = {'png': 'image/png', 'jpg': 'image/jpeg',
                    'jpeg': 'image/jpeg', 'svg': 'image/svg+xml',
                    'gif': 'image/gif'}.get(ext, 'image/png')
            with open(img_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            return f'{prefix}data:{mime};base64,{b64}{suffix}'
        return match.group(0)
    return re.sub(r'(<img\s[^>]*src=["\'])([^"\']+)(["\'])', replacer, html)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report_md')
    parser.add_argument('style_css')
    parser.add_argument('output_dir')
    parser.add_argument('model_name', nargs='?')
    parser.add_argument('--report-kind', choices=('benchmark', 'validation', 'status'),
                        default='benchmark')
    args = parser.parse_args(argv)

    md_path = args.report_md
    css_path = args.style_css
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Derive model name: explicit arg > parent-of-output_dir > "model"
    if args.model_name:
        model_name = args.model_name
    else:
        # output_dir is typically models/{model_name}/reports/ — walk up two levels
        abs_out = os.path.abspath(out_dir)
        model_name = os.path.basename(os.path.dirname(abs_out)) or "model"
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', model_name):
        parser.error('model_name must be a safe single path component')

    base_dir = os.path.dirname(os.path.abspath(md_path))

    with open(md_path, encoding='utf-8') as f:
        md_text = f.read()

    # Strip YAML frontmatter
    md_text = re.sub(r'^---\n.*?\n---\n', '', md_text, count=1, flags=re.DOTALL)

    with open(css_path, encoding='utf-8') as f:
        css = f.read()

    # Convert markdown to HTML
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

    # Wrap in full HTML document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DeepStream {args.report_kind.title()} Report — {html_module.escape(model_name)}</title>
<style>
{css}
@media print {{
  body {{ max-width: 100%; padding: 10px; }}
  img {{ max-width: 100%; page-break-inside: avoid; }}
  table {{ page-break-inside: avoid; }}
  h2 {{ page-break-after: avoid; }}
}}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    # Embed images as base64
    html = embed_images(html, base_dir)

    html_out = os.path.join(out_dir, f'{args.report_kind}_report.html')
    pdf_out = os.path.join(out_dir, f'{args.report_kind}_report_{model_name}.pdf')

    with open(html_out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  HTML: {html_out}")

    # setup.sh owns package installation. Report generation never invokes apt or sudo.
    wkhtmltopdf = find_wkhtmltopdf()
    if not wkhtmltopdf:
        print("  PDF generation blocked: wkhtmltopdf is unavailable. Run setup.sh in "
              "the execution container, rerun preflight once, then retry the report.",
              file=sys.stderr)
        sys.exit(1)

    # Convert to PDF.
    # Intentionally NOT passing --enable-local-file-access: all images have already
    # been converted to base64 data: URIs by embed_images(), and the CSS is inlined
    # in <style>...</style>, so no file:// fetching is needed. Keeping it disabled
    # blocks a CSS/HTML-injection exfil vector if the upstream Markdown ever carries
    # untrusted content (e.g. an HF model card).
    result = subprocess.run(
        [
            wkhtmltopdf,
            '--page-size', 'A4',
            '--margin-top', '15mm',
            '--margin-bottom', '15mm',
            '--margin-left', '15mm',
            '--margin-right', '15mm',
            '--image-quality', '100',
            '--no-outline',
            html_out, pdf_out,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        timeout=300,
    )

    if result.returncode == 0 and os.path.isfile(pdf_out) and os.path.getsize(pdf_out) > 0:
        print(f"  PDF:  {pdf_out}")
    else:
        print(f"  PDF generation failed: {result.stderr[:500]}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
