# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

NVDS_VERSION:=9.2
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CXX:= g++

TARGET_LIB:= libnvds_tritoninferfilter.so

SRCS:= $(wildcard *.cpp)
INCS:= $(wildcard *.h)

OBJS:= $(SRCS:.cpp=.o)

PKGS:= protobuf yaml-cpp

# nvdsinferserver_*.pb.h and the private infer_* headers are picked up from the
# nvdsinferserver build tree; the matching symbols come from
# libnvds_infer_server.so.
CFLAGS+= -I. \
	 -I../.. \
	 -I../../.. \
	 -I../../../nvdsinferserver \
	 -I../../../../../includes \
	 -I../../../../../includes/nvdsinferserver \
	 -I/usr/local/cuda-$(CUDA_VER)/include

CFLAGS+= -fPIC -DHAVE_CONFIG_H -DFOR_PRIVATE -DNDEBUG -std=c++17 -Wall -Werror

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS+= $(shell pkg-config --libs $(PKGS))

LIBS+= -shared -Wl,-no-undefined \
       -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
       -L$(LIB_INSTALL_DIR) -lnvds_3d_common -lnvds_infer_server \
       -ldl -lrt -lm -pthread \
       -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(TARGET_LIB): $(OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(TARGET_LIB)
