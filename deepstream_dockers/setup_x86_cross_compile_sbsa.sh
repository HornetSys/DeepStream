#!/bin/bash
#
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
# limitations under the License


# special docker setup for x86 cross-compile for SBSA
#
# Only needed when building the SBSA image on an x86 host. A native ARM64 host
# needs none of this and builds far faster, since opentelemetry-cpp and gRPC
# are compiled from source.
sudo docker --version


sudo apt-get install qemu binfmt-support qemu-user-static
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
# Verify everything works.
# handle special case if working on shared systems
# On a shared development or CI system restarting docker terminates ALL running
# containers without warning, potentially destroying other users' work or
# in-flight CI jobs.
# Fix: Check whether aarch64 binfmt is already registered before restarting
ls /proc/sys/fs/binfmt_misc/qemu-aarch64 &>/dev/null || sudo systemctl restart docker
# Use Docker buildx
# handle case if this is run a second time and this will not cause an error
sudo docker buildx ls | grep -q mysbsabuilder || docker buildx create --name mysbsabuilder
# Create a new builder instance with support for multiple architectures
sudo docker buildx use mysbsabuilder
# Inspect available platforms and enable ARM64
sudo docker buildx inspect --bootstrap
