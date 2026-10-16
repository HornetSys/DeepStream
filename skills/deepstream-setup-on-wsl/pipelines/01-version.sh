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

# desc: DeepStream SDK + toolchain present (deepstream-app --version-all)  [DOC test 1]
_id=version
_log="$LOGDIR/$_id.log"
echo 'deepstream-app --version-all' > "$LOGDIR/$_id.cmd"
if deepstream-app --version-all >"$_log" 2>&1; then
  _ver=$(grep -aoiE "DeepStreamSDK [0-9.]+" "$_log" | head -1)
  emit "$_id" PASS "${_ver:-deepstream-app present}" ds_version="${_ver:-unknown}"
else
  emit "$_id" FAIL "deepstream-app --version-all failed — see logs/$_id.log"
fi
