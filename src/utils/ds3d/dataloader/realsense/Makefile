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

TARGET_LIB:= libnvds_3d_dataloader_realsense.so

SRCS:= $(wildcard *.cpp)
INCS:= $(wildcard *.h)

OBJS:= $(SRCS:.cpp=.o)

PKGS:= yaml-cpp

# librealsense2 headers are vendored under ds3d/3rdparty; the library is
# resolved at link time against the stub in ds3d/stub/realsense and at run time
# against the real Intel RealSense SDK.
REALSENSE_INC_DIR:= ../../3rdparty/realsense2/include
REALSENSE_STUB_DIR:= ../../stub/realsense

CFLAGS+= -I. \
	 -I../.. \
	 -I../../.. \
	 -I$(REALSENSE_INC_DIR) \
	 -I../../../../../includes \
	 -I/usr/local/cuda-$(CUDA_VER)/include

CFLAGS+= -fPIC -DHAVE_CONFIG_H -DNDEBUG -std=c++17 -fvisibility=hidden \
	 -Wall -Wno-deprecated-declarations

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS+= $(shell pkg-config --libs $(PKGS))

LIBS+= -shared -Wl,-no-undefined \
       -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
       -L$(LIB_INSTALL_DIR) -lnvds_3d_common \
       -L$(REALSENSE_STUB_DIR) -lrealsense2 \
       -ldl -pthread \
       -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(TARGET_LIB)

$(REALSENSE_STUB_DIR)/librealsense2.so:
	$(MAKE) -C $(REALSENSE_STUB_DIR)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(TARGET_LIB): $(OBJS) $(REALSENSE_STUB_DIR)/librealsense2.so Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(TARGET_LIB)
