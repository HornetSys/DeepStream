/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Dynamic loading wrapper for libgstnvdsseimeta and libgstnvcustomhelper
 */

#ifndef __NVHELPER_DYN_H__
#define __NVHELPER_DYN_H__

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize dynamic loading of helper libraries.
 * Call this once at plugin init time.
 */
void nvhelper_dyn_init(void);

/*
 * Cleanup dynamically loaded libraries.
 * Called automatically when the shared library is unloaded.
 */
void nvhelper_dyn_deinit(void);

#ifdef __cplusplus
}
#endif

#endif /* __NVHELPER_DYN_H__ */
