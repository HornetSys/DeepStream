# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Link-time stub for librealsense2. It exports the symbols listed in
# librealsense2.export as empty functions so that the RealSense dataloader can
# be linked without the Intel RealSense SDK installed. At run time the loader
# picks up the real librealsense2.so from the system; the stub is never
# installed.

CC?= gcc

STUB_NAME:= librealsense2.so

all: $(STUB_NAME)

$(STUB_NAME): librealsense2.export Makefile
	awk '{ if (NF && !/^#/) print "void",$$1,"() {}" }' $< | \
	    $(CC) -x c -O0 -fPIC -shared -Wl,-soname=$(STUB_NAME) -Wl,--strip-all -o $@ -

# The stub is a link-time helper only — nothing is installed on the target.
install: $(STUB_NAME)

clean:
	rm -rf $(STUB_NAME)

.PHONY: all install clean
