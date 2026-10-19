#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the checked dense-XYWH parser without a DeepStream installation."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts/deepstream/parsers/yolo_dense_xywh.cpp"


class DenseYoloParserTests(unittest.TestCase):
    def test_channel_first_last_and_bad_shape(self):
        if not shutil.which("g++"):
            self.skipTest("g++ is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "nvdsinfer_custom_impl.h").write_text(
                """#pragma once
#include <vector>
enum NvDsInferDataType { FLOAT, HALF };
struct NvDsInferDims { unsigned int numDims; int d[8]; };
struct NvDsInferLayerInfo { void* buffer; NvDsInferDataType dataType; NvDsInferDims inferDims; };
struct NvDsInferNetworkInfo { unsigned int width; unsigned int height; };
struct NvDsInferParseDetectionParams {
  unsigned int numClassesConfigured;
  std::vector<float> perClassPreclusterThreshold;
};
struct NvDsInferObjectDetectionInfo {
  unsigned int classId; float detectionConfidence; float left; float top;
  float width; float height;
};
#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(name)
""", encoding="utf-8")
            harness = directory / "test.cpp"
            harness.write_text(
                f"""#include <cassert>
#include <cmath>
#include \"{PARSER}\"
int main() {{
  NvDsInferNetworkInfo network{{100, 100}};
  NvDsInferParseDetectionParams params{{2, {{0.25f, 0.25f}}}};
  std::vector<NvDsInferObjectDetectionInfo> objects;
  float chw[] = {{50,30, 50,30, 20,10, 40,10, .9f,.1f, .1f,.8f}};
  NvDsInferLayerInfo first{{chw, FLOAT, {{3, {{1,6,2}}}}}};
  assert(NvDsInferParseDenseYoloXYWH({{first}}, network, params, objects));
  assert(objects.size() == 2);
  assert(objects[0].classId == 0 && objects[1].classId == 1);
  assert(std::fabs(objects[0].left - 40.f) < .001f);
  assert(std::fabs(objects[0].top - 30.f) < .001f);
  float hwc[] = {{50,50,20,40,.9f,.1f, 30,30,10,10,.1f,.8f}};
  NvDsInferLayerInfo second{{hwc, FLOAT, {{2, {{2,6}}}}}};
  assert(NvDsInferParseDenseYoloXYWH({{second}}, network, params, objects));
  assert(objects.size() == 2 && objects[1].classId == 1);
  second.inferDims = {{2, {{2,5}}}};
  assert(!NvDsInferParseDenseYoloXYWH({{second}}, network, params, objects));
  assert(objects.empty());
  second.inferDims = {{2, {{2,6}}}};
  second.dataType = HALF;
  assert(!NvDsInferParseDenseYoloXYWH({{second}}, network, params, objects));
  return 0;
}}
""", encoding="utf-8")
            executable = directory / "parser-test"
            build = subprocess.run(
                ["g++", "-std=c++17", "-O2", "-I", str(directory), str(harness),
                 "-o", str(executable)], text=True, capture_output=True, check=False,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], text=True, capture_output=True, check=False)
            self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
