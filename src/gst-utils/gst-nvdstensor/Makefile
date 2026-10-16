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

CXX:= g++

NVDS_VERSION:=9.2

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -DHAVE_CONFIG_H -std=c++14 -Wall -Werror

CFLAGS+= -I. \
	 -I../../../includes \
	 -I/usr/local/cuda-$(CUDA_VER)/include

PKGS:= glib-2.0 gobject-2.0 gstreamer-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

SRCS:= gstnvdstensor.cpp nvdstensor_common.c nvdstensor.c
LIB:= libnvdsgst_tensor.so

LIBS:= -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
	-ldl \
	-Wl,-rpath,$(LIB_INSTALL_DIR)
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(LIB)

$(LIB): $(SRCS) Makefile
	$(CXX) -o $@ $(SRCS) $(CFLAGS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(LIB)
