###############################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program. If not, see
# <http://www.gnu.org/licenses/>.
###############################################################################

SO_NAME := libnvv4l2.so

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
ifneq ($(TARGET_DEVICE),x86_64)
  $(error "This Makefile builds libv4l2 for x86_64 only")
endif

CC := gcc

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

SRCS := libv4l2.c log.c v4l2convert.c v4l2-plugin.c

INCLUDES += -I. -I../include

OBJS := $(SRCS:.c=.o)

CFLAGS := -fPIC -DLIBV4L2_PLUGIN_DIR_PATH_X86

LDFLAGS := -Wl,--no-undefined -Wl,-soname,libv4l2.so.0 \
	-L../libv4lconvert -L$(LIB_INSTALL_DIR) -Wl,-rpath,$(LIB_INSTALL_DIR)

LIBS := -lnvv4lconvert -ldl

all: $(SO_NAME)

%.o: %.c
	$(CC) -c $< $(CFLAGS) $(INCLUDES) -o $@

$(SO_NAME): $(OBJS)
	$(CC) -shared -o $(SO_NAME) $(OBJS) $(LIBS) $(LDFLAGS)

.PHONY: install
install: $(SO_NAME)
	cp -vp $(SO_NAME) $(LIB_INSTALL_DIR)
	ln -sfn $(SO_NAME) $(LIB_INSTALL_DIR)/libv4l2.so.0

.PHONY: clean
clean:
	rm -rf $(OBJS) $(SO_NAME)
