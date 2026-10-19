################################################################################
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
################################################################################

# Accuracy Validation Mode

This plugin includes optional accuracy validation code that can be enabled for testing and debugging.

## What It Does

When enabled, the plugin:
1. **Prints debug messages** for the first 3 frames (first 2 objects per frame)
   - Success: `[C-PLUGIN] Frame X, Obj Y: Attached real embedding...`
   - Failure: `[C-PLUGIN-ERROR] Frame X, Obj Y: Failed to attach embedding`

2. **Saves first detected object's data** (frame 1, object 0):
   - **Embedding vector** → `/tmp/test_data/deepstream_embedding.txt`
   - **Bounding box coordinates** → `/tmp/test_data/bbox_coords.txt`

This allows you to:
1. Verify embeddings are being generated correctly
2. Compare DeepStream embeddings with reference implementations
3. Extract exact bounding boxes for creating test cases
4. Debug embedding attachment failures

## How to Enable

### Option 1: Edit Makefile (Recommended)

Uncomment this line in `Makefile`:
```makefile
# Uncomment to enable accuracy validation code
CFLAGS += -DENABLE_ACCURACY_VALIDATION
```

Then rebuild:
```bash
make clean && make -j$(nproc)
```

### Option 2: Command Line

Build with the flag directly:
```bash
make clean
make CFLAGS+="-DENABLE_ACCURACY_VALIDATION" -j$(nproc)
```

### Option 3: Edit Source

Uncomment this line in `gstnvdsvisionencoder.c` (line ~23):
```c
// #define ENABLE_ACCURACY_VALIDATION
```

Then rebuild.

## Output Files

When enabled, the plugin creates:

**`/tmp/test_data/deepstream_embedding.txt`:**
```
0.4543457031
-0.1257324219
0.3071289062
...
(1024 lines total)
```

**`/tmp/test_data/bbox_coords.txt`:**
```
left=1020.23
top=291.06
width=121.41
height=53.26
```

## Usage Example

```bash
# 1. Enable validation mode (uncomment in Makefile)
# 2. Rebuild
make clean && make -j$(nproc)
cp libgstnvdsvisionencoder.so /opt/nvidia/deepstream/deepstream/lib/gst-plugins/

# 3. Run DeepStream pipeline
gst-launch-1.0 ... nvdsvisionencoder ... ! fakesink

# 4. Check outputs
cat /tmp/test_data/deepstream_embedding.txt
cat /tmp/test_data/bbox_coords.txt

# 5. Compare with reference
python3 compare_accuracy.py
```

## Performance Impact

### When Disabled (Production):
- ✅ **Zero overhead** - Code is completely compiled out
- ✅ **No logging** - No string formatting or console output
- ✅ **No file I/O** - No disk writes

### When Enabled (Testing):
- **Debug logging:** Minimal (only first 3 frames, 2 objects each = ~6 messages)
- **File writes:** Once per run (first frame, first object = ~14KB total)
- **Overall impact:** Negligible

## Production Builds

For production, keep this feature **disabled** (commented out in Makefile):
```makefile
# CFLAGS += -DENABLE_ACCURACY_VALIDATION  # Disabled for production
```

## Validation Results

With this feature, we validated:
- ✅ **Cosine Similarity: 0.9628** (batch-16 vs reference)
- ✅ Preprocessing correctness ([0,1] normalization)
- ✅ Batched inference accuracy
