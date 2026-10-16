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

NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc
CXX:=g++

NVDS_VERSION:=9.2
GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

PKGS:=gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS:=-fPIC -std=c++11 -DDS_VERSION=\"9.2.0\" \
        -I. -I../../../includes -I/usr/local/cuda-$(CUDA_VER)/include
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
NVCC_CFLAGS:=-I. -I../../../includes -I/usr/local/cuda-$(CUDA_VER)/include

SRCS:=gstnvdsframeselector.cpp motion_detector.cpp sad_cpu.cpp sad_kernel.cu
OBJS:=gstnvdsframeselector.o motion_detector.o sad_cpu.o sad_kernel.o
LIB:=libnvdsgst_frameselector.so

LIBS:=-shared -Wl,-no-undefined \
      -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lcuda \
      -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsbufferpool -lnvds_meta \
      -lnvdsgst_meta -lnvbufsurface -lnvbufsurftransform -lnvds_nvtxhelper \
      -Wl,-rpath,$(LIB_INSTALL_DIR) -ldl
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu Makefile
	$(NVCC) -c -o $@ $(NVCC_CFLAGS) --compiler-options '-fPIC' $<

$(LIB): $(OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
