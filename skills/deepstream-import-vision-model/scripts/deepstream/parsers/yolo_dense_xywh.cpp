// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Decode an already-regressed dense YOLO head [4+C, N] or [N, 4+C].
// This parser does not decode YOLOX grids, objectness, fused NMS, or logits.
// The caller must inspect the exact ONNX tensor, set C in nvinfer, and validate
// real-frame boxes before treating this type-level parser as compatible.

#include "nvdsinfer_custom_impl.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

extern "C" bool NvDsInferParseDenseYoloXYWH(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects) {
  objects.clear();
  if (layers.size() != 1 || params.numClassesConfigured == 0 ||
      params.numClassesConfigured > 1000 || network.width == 0 || network.height == 0 ||
      params.perClassPreclusterThreshold.size() < params.numClassesConfigured) {
    std::cerr << "dense YOLO parser: invalid layer/class/network contract\n";
    return false;
  }

  const auto& layer = layers.front();
  if (layer.buffer == nullptr || layer.dataType != FLOAT) {
    std::cerr << "dense YOLO parser: output must be a non-null FP32 tensor\n";
    return false;
  }
  const auto& dims = layer.inferDims;
  unsigned int first = 0;
  if (dims.numDims == 2) {
    first = 0;
  } else if (dims.numDims == 3 && dims.d[0] == 1) {
    first = 1;
  } else {
    std::cerr << "dense YOLO parser: expected rank 2 or singleton-batch rank 3\n";
    return false;
  }
  const unsigned int channels = params.numClassesConfigured + 4;
  const unsigned int rows = dims.d[first];
  const unsigned int cols = dims.d[first + 1];
  const bool channel_first = rows == channels && cols != channels;
  const bool channel_last = cols == channels && rows != channels;
  if ((!channel_first && !channel_last) || rows == 0 || cols == 0 ||
      rows > 200000 || cols > 200000) {
    std::cerr << "dense YOLO parser: output shape does not match 4+C channels\n";
    return false;
  }

  const unsigned int anchors = channel_first ? cols : rows;
  const float* values = static_cast<const float*>(layer.buffer);
  const auto at = [&](unsigned int anchor, unsigned int channel) -> float {
    return channel_first ? values[channel * anchors + anchor]
                         : values[anchor * channels + channel];
  };
  const float max_x = static_cast<float>(network.width);
  const float max_y = static_cast<float>(network.height);
  for (unsigned int anchor = 0; anchor < anchors; ++anchor) {
    const float cx = at(anchor, 0);
    const float cy = at(anchor, 1);
    const float width = at(anchor, 2);
    const float height = at(anchor, 3);
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(width) ||
        !std::isfinite(height) || width <= 0.0f || height <= 0.0f) {
      continue;
    }
    unsigned int best_class = 0;
    float best_score = -1.0f;
    for (unsigned int class_id = 0; class_id < params.numClassesConfigured; ++class_id) {
      const float score = at(anchor, class_id + 4);
      if (std::isfinite(score) && score > best_score) {
        best_score = score;
        best_class = class_id;
      }
    }
    if (best_score < params.perClassPreclusterThreshold[best_class] ||
        best_score > 1.0f) {
      continue;
    }
    const float left = std::clamp(cx - width * 0.5f, 0.0f, max_x);
    const float top = std::clamp(cy - height * 0.5f, 0.0f, max_y);
    const float right = std::clamp(cx + width * 0.5f, 0.0f, max_x);
    const float bottom = std::clamp(cy + height * 0.5f, 0.0f, max_y);
    if (right - left < 1.0f || bottom - top < 1.0f) {
      continue;
    }
    NvDsInferObjectDetectionInfo object{};
    object.classId = best_class;
    object.detectionConfidence = best_score;
    object.left = left;
    object.top = top;
    object.width = right - left;
    object.height = bottom - top;
    objects.push_back(object);
  }
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseDenseYoloXYWH);
