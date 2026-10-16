/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */

#include "wsl_utils.h"

bool is_running_in_WSL(void)
{
  static volatile bool verified = false;
  static volatile bool ret = false;

  if (!verified) {
    verified = true;
    FILE *versionFile = fopen("/proc/version", "r");
    if (versionFile != NULL) {
      char versionInfo[512];
      if (fgets(versionInfo, sizeof(versionInfo), versionFile) != NULL) {
        for (int i=0; versionInfo[i] != '\0'; i++) {
          versionInfo[i] = tolower((unsigned char)versionInfo[i]);
        }
        if (strstr(versionInfo, "microsoft") != NULL) {
          /* Yes, Running inside WSL */
          ret = true;
        }
      }
      fclose(versionFile);
    } else {
      printf("ERROR: opening /proc/version failed\n");
    }
  }

  return ret;
}
