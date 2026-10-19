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

CXX:= g++

NVDS_VERSION:=9.2

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -std=c++11 -Wall

CFLAGS+= -I. \
	 -I../../../includes

PKGS:= glib-2.0 gobject-2.0 json-glib-1.0

SRCS:= nv_aisle_csvparser.cpp nv_spot_csvparser.cpp

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIB:= libnvds_csvparser.so

LIBS:= -shared -Wl,-no-undefined \
	-lpthread \
	-Wl,-rpath,$(LIB_INSTALL_DIR)
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(LIB)

$(LIB): $(SRCS) Makefile
	$(CXX) -o $@ $(SRCS) $(CFLAGS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(LIB)
