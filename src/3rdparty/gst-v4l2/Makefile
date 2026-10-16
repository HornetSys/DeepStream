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

SO_NAME := libgstnvvideo4linux2.so

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
ifneq ($(TARGET_DEVICE),x86_64)
  $(error "This Makefile builds gst-v4l2 for x86_64 only")
endif

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/
CFLAGS:= -DUSE_V4L2_TARGET_NV_CODECSDK=1 -DUSE_V4L2_TARGET_NV_X86=1 -DUSE_V4L2_GST_HEADER_VER_1_8

LIBS:= -lnvbufsurface -lnvbufsurftransform -lnvv4l2 -ldl
SRCS := $(wildcard *.c)

INCLUDES += -I. \
	-I../../../includes \
	-I../../../includes/libv4l2/include \
	-I../../../includes/libv4l2/include/linux

PKGS := gstreamer-1.0 \
	gstreamer-base-1.0 \
	gstreamer-video-1.0 \
	gstreamer-allocators-1.0 \
	glib-2.0

OBJS := $(SRCS:.c=.o)

CFLAGS += -fPIC \
	-DEXPLICITLY_ADDED=1 \
        -DGETTEXT_PACKAGE=1 \
        -DHAVE_LIBV4L2=1 \
        -DUSE_V4L2_TARGET_NV=1

CFLAGS += `pkg-config --cflags $(PKGS)`

LDFLAGS = -Wl,--no-undefined -L$(LIB_INSTALL_DIR) -Wl,-rpath,$(LIB_INSTALL_DIR)

LIBS += `pkg-config --libs $(PKGS)`

all: $(SO_NAME)

%.o: %.c
	$(CC) -c $< $(CFLAGS) $(INCLUDES) -o $@

$(SO_NAME): $(OBJS)
	$(CC) -shared -o $(SO_NAME) $(OBJS) $(LIBS) $(LDFLAGS)

.PHONY: install
install: $(SO_NAME)
	cp -vp $(SO_NAME) $(GST_INSTALL_DIR)

.PHONY: clean
clean:
	rm -rf $(OBJS) $(SO_NAME)
