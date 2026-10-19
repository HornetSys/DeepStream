# SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Steps to compile the "gst-nvvideo4linux2" sources natively.

x86_64 only. The Makefile stops on any other architecture, including Jetson.
The sources are already in this directory; there is no separate source archive
to download.

libgstnvvideo4linux2.so provides nvv4l2decoder and links -lnvv4l2. Build and
install libnvv4lconvert.so and libnvv4l2.so first, from
../v4l2/libv4lconvert and ../v4l2/libv4l2 (see those READMEs). Both install
under /opt/nvidia/deepstream/deepstream/lib/.

1) Install gstreamer related packages on the target:

	sudo apt-get install libgstreamer1.0-dev \
		gstreamer1.0-plugins-base \
		gstreamer1.0-plugins-good \
		libgstreamer-plugins-base1.0-dev \
		libv4l-dev \
		libegl1-mesa-dev

2) From this directory, build and install "libgstnvvideo4linux2.so":

	make
	sudo make install

   make install copies libgstnvvideo4linux2.so to
   /opt/nvidia/deepstream/deepstream/lib/gst-plugins/
   (GST_INSTALL_DIR). To install somewhere else:

	make install GST_INSTALL_DIR=<dir>/

   The Makefile does not read DEST_DIR.

3) Verify:

	gst-inspect-1.0 nvv4l2decoder
