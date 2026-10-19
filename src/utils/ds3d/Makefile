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

# Aggregator for the DS3D component libraries. "common", "glprocess" and the
# realsense stub are listed first because every other subdirectory links
# against them.
SUBDIRS:= \
	common \
	glprocess \
	stub/realsense \
	dataloader/realsense \
	dataloader/lidarsource \
	dataloader/depthsource \
	datafilter/depth2point \
	datafilter/lidarinference \
	datafilter/lidar_preprocess \
	datafilter/alignment \
	datafilter/sensorfusion_cluster \
	datarender/gl3d \
	datarender/gl3d_ensemble \
	datarender/lidarsink \
	databridge/video \
	datamixer/multisensor \
	inference_custom_lib \
	gst

all install clean:
	@for dir in $(SUBDIRS); do \
	    $(MAKE) -C $$dir $@ || exit 1; \
	done

.PHONY: all install clean
