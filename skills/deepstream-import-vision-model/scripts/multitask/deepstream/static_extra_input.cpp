// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Static extra-tensor initializer for a fixed-pair DeepStream diagnostic.
// The SDK calls this hook once, so it MUST NOT be used for changing stereo
// frames, per-frame prompts, tracking memory, or synchronized video pairs.
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferInitializeInputLayers(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo&, unsigned int max_batch_size) {
  const char* name = std::getenv("DS_STATIC_EXTRA_INPUT_NAME");
  const char* path = std::getenv("DS_STATIC_EXTRA_INPUT_FILE");
  if (!name || !*name || !path || !*path || layers.size() != 1 ||
      max_batch_size == 0 || max_batch_size > 512) {
    std::cerr << "Static extra input requires one named layer, one file, and a bounded batch\n";
    return false;
  }
  const auto& layer = layers.front();
  if (!layer.layerName || name != std::string(layer.layerName) ||
      layer.dataType != FLOAT || !layer.buffer ||
      layer.inferDims.numElements == 0 ||
      layer.inferDims.numElements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "Static extra input name, FP32 dtype, or dimensions do not match\n";
    return false;
  }
  const size_t bytes = static_cast<size_t>(layer.inferDims.numElements) * sizeof(float);
  if (bytes > std::numeric_limits<size_t>::max() / max_batch_size ||
      bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) return false;
  const std::filesystem::path input(path);
  std::error_code error;
  if (!std::filesystem::is_regular_file(input, error) || error ||
      std::filesystem::is_symlink(input, error) || error ||
      std::filesystem::file_size(input, error) != bytes || error) {
    std::cerr << "Static extra input file missing or size does not match layer shape\n";
    return false;
  }
  std::ifstream stream(input, std::ios::binary);
  if (!stream.read(static_cast<char*>(layer.buffer), static_cast<std::streamsize>(bytes))) {
    std::cerr << "Static extra input could not be read completely\n";
    return false;
  }
  const auto* values = static_cast<const float*>(layer.buffer);
  for (size_t i = 0; i < layer.inferDims.numElements; ++i) {
    if (!std::isfinite(values[i])) {
      std::cerr << "Static extra input contains a non-finite value\n";
      return false;
    }
  }
  for (unsigned int batch = 1; batch < max_batch_size; ++batch) {
    std::memcpy(static_cast<char*>(layer.buffer) + batch * bytes, layer.buffer, bytes);
  }
  return true;
}
