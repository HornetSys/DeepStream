# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
CUDA_VER?=
# Require CUDA_VER for every goal except a clean-only invocation.
# A bare `make` has empty MAKECMDGOALS and still checks.
ifneq ($(filter-out clean,$(MAKECMDGOALS)),)
  _DS_NEED_CUDA_VER := 1
else ifeq ($(MAKECMDGOALS),)
  _DS_NEED_CUDA_VER := 1
endif
ifeq ($(_DS_NEED_CUDA_VER),1)
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
endif

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
ifneq ($(TARGET_DEVICE),x86_64)
  $(error "This Makefile builds gst-nvunixfd for x86_64 only")
endif

SO_NAME := libgstnvunixfd.so

CC := gcc

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

CUDA_HOME:= /usr/local/cuda-$(CUDA_VER)
CFLAGS:=
LIBS:= -lnvbufsurface -lnvbufsurftransform -lpthread -ldl

SRCS := $(wildcard *.c)

INCLUDES += -I. -I../../../includes
INCLUDES += -I$(CUDA_HOME)/include

PKGS := gstreamer-1.0 \
	gstreamer-base-1.0 \
	gstreamer-video-1.0 \
	gstreamer-allocators-1.0 \
	glib-2.0 \
	gio-2.0 \
	gio-unix-2.0

OBJS := $(SRCS:.c=.o)

CFLAGS += -fPIC \
	-DEXPLICITLY_ADDED=1 \
	-DGETTEXT_PACKAGE=1 \
	-DHAVE_IPC_TARGET_NV=1

CFLAGS += $(shell pkg-config --cflags $(PKGS))

LDFLAGS = -Wl,--no-undefined -L$(LIB_INSTALL_DIR) -Wl,-rpath,$(LIB_INSTALL_DIR)

LIBS += $(shell pkg-config --libs $(PKGS))

LIBS += -L$(CUDA_HOME)/lib64 -lcuda -lcudart

all: $(SO_NAME)

%.o: %.c
	@echo "Compiling: $(CC) $(CFLAGS) $<"
	$(CC) -c $< $(CFLAGS) $(INCLUDES) -o $@

$(SO_NAME): $(OBJS)
	$(CC) -shared -o $(SO_NAME) $(OBJS) $(LIBS) $(LDFLAGS)

.PHONY: install
install: $(SO_NAME)
	cp -vp $(SO_NAME) $(GST_INSTALL_DIR)

.PHONY: clean
clean:
	rm -rf $(OBJS) $(SO_NAME)
