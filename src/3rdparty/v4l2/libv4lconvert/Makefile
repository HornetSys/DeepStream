###############################################################################
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License, version 2.1, as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program. If not, see
# <http://www.gnu.org/licenses/>.
#
###############################################################################

SO_NAME := libnvv4lconvert.so

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
ifneq ($(TARGET_DEVICE),x86_64)
  $(error "This Makefile builds libv4lconvert for x86_64 only")
endif

CC := gcc

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

SRCS := libv4lconvert.c tinyjpeg.c sn9c10x.c sn9c20x.c pac207.c  mr97310a.c \
	flip.c crop.c jidctflt.c spca561-decompress.c \
	rgbyuv.c sn9c2028-decomp.c spca501.c sq905c.c bayer.c hm12.c \
	stv0680.c cpia1.c se401.c jpgl.c jpeg.c jl2005bcd.c helper.c \
	$(wildcard processing/*.c) \
	$(wildcard control/*.c)

INCLUDES += -I. -I../include -I./control -I./processing

OBJS := $(SRCS:.c=.o)

CFLAGS := -fPIC -DLIBV4L2_PLUGIN_DIR_PATH_X86

LIBS := -lrt -lm

LDFLAGS := -Wl,--no-undefined -Wl,-soname,libv4lconvert.so.0 \
	-L$(LIB_INSTALL_DIR) -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(SO_NAME)

%.o: %.c
	$(CC) -c $< $(CFLAGS) $(INCLUDES) -o $@

$(SO_NAME): $(OBJS)
	$(CC) -shared -o $(SO_NAME) $(OBJS) $(LIBS) $(LDFLAGS)

.PHONY: install
install: $(SO_NAME)
	cp -vp $(SO_NAME) $(LIB_INSTALL_DIR)
	ln -sfn $(SO_NAME) $(LIB_INSTALL_DIR)/libv4lconvert.so.0

.PHONY: clean
clean:
	rm -rf $(OBJS) $(SO_NAME)
