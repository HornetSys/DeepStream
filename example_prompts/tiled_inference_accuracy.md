<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->

Build a Python DeepStream `pyservicemaker` application that compares ordinary full-frame detection
with overlapping tiled detection on the same video and model. Let me choose the video, model
configuration, tile grid, overlap, frame limit, and output directory; use the shipped sample video
and primary-inference configuration when I do not provide them.

Run a baseline and tiled mode over the same frames. For each mode, write an annotated MP4 and JSON
summary with processed-frame count, detection counts, small-object counts, and end-to-end FPS. Merge
duplicate tiled detections by class using configurable overlap thresholds, and print the baseline,
tiled, and percentage deltas.

Include a simple launcher, focused tests for tile geometry and duplicate merging, and a README with
setup and run instructions. Clearly state that detection-count changes are only a recall proxy:
use labeled-data precision/recall or mAP to make accuracy claims.
