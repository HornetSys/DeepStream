################################################################################
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
################################################################################

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

CC:= g++

CFLAGS:= -Wall -std=c++14 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I../../../../../includes \
         -I /usr/local/cuda-$(CUDA_VER)/include \
         -I/opt/nvidia/deepstream/deepstream/sources/includes

LIBS:= \
       -L/opt/nvidia/deepstream/deepstream/lib/ \
       -lnvdsgst_helper \
       -lnvdsgst_meta \
       -lnvds_meta \
       -lnvds_inferutils
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsinfer_custom_impl_yolo.cpp

TARGET_LIB:= libnvdsinfer_custom_impl_yolo.so

PKGS:= glib-2.0
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
