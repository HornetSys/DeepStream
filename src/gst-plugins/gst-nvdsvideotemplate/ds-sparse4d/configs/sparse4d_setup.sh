#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SPARSE4D_CONFIG_PATH="${SPARSE4D_CONFIG_PATH:-/opt/data/ds-configurator/}"
SHARED_CONFIG_FILE_PATH="${SPARSE4D_CONFIG_PATH}config.yaml"
GPU_ARCH=$(nvidia-smi -q | grep "Product Architecture" | cut -d: -f2 | tr -d ' ' | head -1)
CONFIG_FILE_PATH="/opt/nvidia/deepstream/deepstream/sources/sparse4d/configs/config.yaml"
PREPROCESS_FILE_PATH="/opt/nvidia/deepstream/deepstream/sources/sparse4d/configs/config_preprocess.txt"
NETWORK_WIDTH="${NETWORK_WIDTH:-0}"
NETWORK_HEIGHT="${NETWORK_HEIGHT:-0}"
MSDA_PLUGIN_PATH="${MSDA_PLUGIN_PATH:-/opt/nvidia/deepstream/deepstream/sources/sparse4d/libmsda_fp16.so}"

fetch_yaml_property_value() {
  yaml_file="$1"
  property="$2"

  if [[ -z "$yaml_file" || -z "$property" ]]; then
    echo "[ERROR] Usage: fetch_and_print_yaml_property_value <yaml_file> <property> [fetch_only]" >&2
    return 1
  fi

  # Extract the property value using grep and awk, remove comments after #
  value=$(grep "^[[:space:]]*${property}:" "$yaml_file" | awk -F':' '{print $2}' | sed 's/#.*$//' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

  if [[ -n "$value" ]]; then
    echo "$value"
  else
    echo "[ERROR] Property '${property}' not found in '${yaml_file}'" >&2
    return 1
  fi
}

# Function to update a property in a YAML file
update_yaml() {
  yaml_file="$1"
  property="$2"
  new_value="$3"

  if [[ -z "$yaml_file" || -z "$property" || -z "$new_value" ]]; then
    echo "[ERROR] Usage: update_yaml <yaml_file> <property> <new_value>" >&2
    return 1
  fi

  # Escape special characters in the property name for sed
  escaped_property=$(echo "$property" | sed 's/[]\/$*.^[]/\\&/g')

  # Use sed to update the property value
  sed -i "s|^\(\s*${escaped_property}:\s*\).*|\1${new_value}|" "$yaml_file"

  if [[ $? -eq 0 ]]; then
    echo "[INFO] Updated $property to $new_value in $yaml_file."
  else
    echo "[ERROR] Failed to update $property in $yaml_file." >&2
    return 1
  fi
}

# Function to validate if a file exists and is readable
# Usage: validate_file_exists "OCRNet dictionary file" "${DICT_FILE}"
validate_file_exists() {
  file_description="$1"
  file_path="$2"

  if [[ -z "$file_description" || -z "$file_path" ]]; then
    echo "[ERROR] Usage: validate_file_exists <file_description> <file_path>" >&2
    return 1
  fi

  if [[ -f "${file_path}" && -r "${file_path}" ]]; then
    echo "[INFO] ${file_description}: ${file_path}"
    return 0
  elif [[ -f "${file_path}" && ! -r "${file_path}" ]]; then
    echo "[ERROR] ${file_description} '${file_path}' exists but does not have read permission." >&2
    exit 1
  else
    echo "[ERROR] ${file_description} '${file_path}' does not exist." >&2
    exit 1
  fi
}

# Function to round to even number within bounds
round_to_even() {
    local x=$1
    local dim_value=$2
    local rounded_value

    # Convert to integer
    x_int=$(printf "%.0f" $x)

    # Check if even
    if [[ $((x_int % 2)) -eq 0 ]]; then
        rounded_value=$x_int
    else
        rounded_value=$((x_int + 1))
    fi

    # Check bounds if dim_value is provided and greater than 0
    if [[ "$dim_value" -gt 0 && "$rounded_value" -gt "$dim_value" ]]; then
        if [[ $((x_int % 2)) -eq 0 ]]; then
            rounded_value=$x_int
        else
            rounded_value=$((x_int - 1))
        fi
    fi

    echo $rounded_value
    return 0
}

update_preprocessed_dims() {
  NETWORK_WIDTH="$1"
  NETWORK_HEIGHT="$2"
  CONFIG_FILE_PATH="$3"
  PREPROCESS_FILE_PATH="$4"
  NUM_SENSORS="$5" # Number of sensors [Deepstream batch size]
  ORIGINAL_HEIGHT=1080
  ORIGINAL_WIDTH=1920

  # Helper function for integer division with rounding
  # Usage: divide_round numerator denominator scale
  divide_round() {
    local num=$1
    local den=$2
    local scale=${3:-6}  # default scale of 6
    # Calculate 10^scale using a loop for POSIX compatibility
    local factor=1
    local i=0
    while [[ $i -lt $scale ]]; do
      factor=$((factor * 10))
      i=$((i + 1))
    done
    local result=$(( (num * factor + den/2) / den ))
    echo $result
    return 0
  }

  # Calculate resize factor using integer arithmetic (scale of 6 for precision)
  # RESIZE_H = NETWORK_HEIGHT / ORIGINAL_HEIGHT
  # RESIZE_W = NETWORK_WIDTH / ORIGINAL_WIDTH
  RESIZE_H_SCALED=$(divide_round $NETWORK_HEIGHT $ORIGINAL_HEIGHT 6)
  RESIZE_W_SCALED=$(divide_round $NETWORK_WIDTH $ORIGINAL_WIDTH 6)

  # RESIZE = max(RESIZE_H, RESIZE_W)
  if [[ $RESIZE_H_SCALED -gt $RESIZE_W_SCALED ]]; then
    RESIZE_SCALED=$RESIZE_H_SCALED
  else
    RESIZE_SCALED=$RESIZE_W_SCALED
  fi

  # For display purposes, convert back to decimal format
  RESIZE_INT=$((RESIZE_SCALED / 1000000))
  RESIZE_FRAC=$((RESIZE_SCALED % 1000000))
  RESIZE=$(printf "%d.%06d" $RESIZE_INT $RESIZE_FRAC | sed 's/0*$//')

  # Calculate resize dimensions
  # SCALED_WIDTH = ORIGINAL_WIDTH * RESIZE
  # SCALED_HEIGHT = ORIGINAL_HEIGHT * RESIZE
  SCALED_WIDTH=$(( (ORIGINAL_WIDTH * RESIZE_SCALED + 500000) / 1000000 ))
  SCALED_HEIGHT=$(( (ORIGINAL_HEIGHT * RESIZE_SCALED + 500000) / 1000000 ))

  # Calculate crop parameters
  # For crop_w, we need max(0, (scaled_width - network_input_width) / 2)
  CROP_W_CALC=$(( (SCALED_WIDTH - NETWORK_WIDTH) / 2 ))
  if [[ $CROP_W_CALC -lt 0 ]]; then
      CROP_W_CALC=0
  fi
  CROP_W=$(round_to_even $CROP_W_CALC $((SCALED_WIDTH - NETWORK_WIDTH)))

  # For crop_h
  CROP_H_CALC=$((SCALED_HEIGHT - NETWORK_HEIGHT))
  CROP_H=$(round_to_even $CROP_H_CALC $CROP_H_CALC)

  # Calculate ROI parameters
  # X1 = CROP_W / RESIZE
  # Y1 = CROP_H / RESIZE
  # W = NETWORK_WIDTH / RESIZE
  # H = NETWORK_HEIGHT / RESIZE
  if [[ "$RESIZE_SCALED" -ne 0 ]]; then
    X1_CALC=$(( (CROP_W * 1000000 + RESIZE_SCALED/2) / RESIZE_SCALED ))
    Y1_CALC=$(( (CROP_H * 1000000 + RESIZE_SCALED/2) / RESIZE_SCALED ))
    W_CALC=$(( (NETWORK_WIDTH * 1000000 + RESIZE_SCALED/2) / RESIZE_SCALED ))
    H_CALC=$(( (NETWORK_HEIGHT * 1000000 + RESIZE_SCALED/2) / RESIZE_SCALED ))
  else
    X1_CALC=0
    Y1_CALC=0
    W_CALC=0
    H_CALC=0
  fi

  X1=$(round_to_even $X1_CALC $ORIGINAL_WIDTH)
  Y1=$(round_to_even $Y1_CALC $ORIGINAL_HEIGHT)
  W=$(round_to_even $W_CALC $ORIGINAL_WIDTH)
  H=$(round_to_even $H_CALC $ORIGINAL_HEIGHT)

  # Update config.yaml
  sed -i "s/preprocessed_height:.*/preprocessed_height: $NETWORK_HEIGHT/" $CONFIG_FILE_PATH
  sed -i "s/preprocessed_width:.*/preprocessed_width: $NETWORK_WIDTH/" $CONFIG_FILE_PATH
  sed -i "/aug_configs:/,/^[^ ]/ {
    s/resize:.*/resize: $RESIZE/
    s/resize_dims:.*/resize_dims: [$SCALED_WIDTH, $SCALED_HEIGHT]/
    s/crop:.*/crop: [$CROP_W, $CROP_H, $NETWORK_WIDTH, $NETWORK_HEIGHT]/
  }" $CONFIG_FILE_PATH

  # Update config_preprocess.txt
  sed -i "s/processing-width=.*/processing-width=$NETWORK_WIDTH/" $PREPROCESS_FILE_PATH
  sed -i "s/processing-height=.*/processing-height=$NETWORK_HEIGHT/" $PREPROCESS_FILE_PATH
  sed -i "s/network-input-shape=.*/network-input-shape=$NUM_SENSORS;3;$NETWORK_HEIGHT;$NETWORK_WIDTH/" $PREPROCESS_FILE_PATH
  sed -i "s/roi-params-src-0=.*/roi-params-src-0=$X1;$Y1;$W;$H/" $PREPROCESS_FILE_PATH

  echo "[INFO] Updated preprocessed dimensions in config.yaml and config_preprocess.txt"
  return 0
}

# Check if DS-Configurator path exists
if [[ -d "${SPARSE4D_CONFIG_PATH}" ]]; then
  echo "[INFO] DS-Configurator path: ${SPARSE4D_CONFIG_PATH}"
else
  echo "[WARN] DS-Configurator path does not exist: ${SPARSE4D_CONFIG_PATH}"
fi

# Check if the shared config file exists
if [[ -f "${SHARED_CONFIG_FILE_PATH}" ]]; then
  # Check if the shared config file is readable
  if [[ -r "${SHARED_CONFIG_FILE_PATH}" ]]; then
    # Copy config file to destination directory
    cp "${SHARED_CONFIG_FILE_PATH}" "${CONFIG_FILE_PATH}"
    echo "[INFO] Copied Shared Sparse4D Config file: ${SHARED_CONFIG_FILE_PATH} to ${CONFIG_FILE_PATH}"
    echo "[INFO] Sparse4D Config file: ${CONFIG_FILE_PATH}"
  else
    echo "[WARN] Shared Sparse4D Config file exists but is not accessible (permission issue): ${SHARED_CONFIG_FILE_PATH}"
    echo "[INFO] Sparse4D Config file: ${CONFIG_FILE_PATH}"
  fi
else
  echo "[WARN] Shared Sparse4D Config file doesn't exist: ${SHARED_CONFIG_FILE_PATH}"
  echo "[INFO] Sparse4D Config file: ${CONFIG_FILE_PATH}"
fi

CALIB_FILE_PATH=$(fetch_yaml_property_value "${CONFIG_FILE_PATH}" "calib_file_path")
# Validate calibration file exists and is readable
validate_file_exists "Calibration file" "${CALIB_FILE_PATH}"

if [[ -z "$ONNX_MODEL_PATH" ]]; then
  # Update onnx_model_path from the config file if ONNX_MODEL_PATH is not set
  ONNX_MODEL_PATH=$(fetch_yaml_property_value "${CONFIG_FILE_PATH}" "onnx_file")
else
  # Update onnx_model_path in the config file if ONNX_MODEL_PATH is set
  update_yaml "${CONFIG_FILE_PATH}" "onnx_file" "${ONNX_MODEL_PATH}"
fi
# Validate Sparse4D ONNX model file exists and is accessible
validate_file_exists "Sparse4D ONNX model file" "${ONNX_MODEL_PATH}"

if [[ -z "$MODEL_LABELS_PATH" ]]; then
  # Update model_labels_path from the config file if MODEL_LABELS_PATH is not set
  MODEL_LABELS_PATH=$(fetch_yaml_property_value "${CONFIG_FILE_PATH}" "labels_file")
else
  # Update model_labels_path in the config file if MODEL_LABELS_PATH is set
  update_yaml "${CONFIG_FILE_PATH}" "labels_file" "${MODEL_LABELS_PATH}"
fi
# Validate model label file exists and is readable
validate_file_exists "Sparse4D Model labels file" "${MODEL_LABELS_PATH}"
echo "[INFO] Sparse4D Labels: $(cat "${MODEL_LABELS_PATH}")"

# Fetch the engine file path from the config file
ENGINE_FILE_PATH=$(fetch_yaml_property_value "${CONFIG_FILE_PATH}" "engine_file")

# Check if the force_engine_rebuild flag is set to True
# If it is, remove the local engine file
FORCE_ENGINE_REBUILD=$(fetch_yaml_property_value "${CONFIG_FILE_PATH}" "force_engine_rebuild")
if [[ "${FORCE_ENGINE_REBUILD}" == "True" ]]; then
  echo "[INFO] Force engine rebuild is set to True, removing local engine file ${ENGINE_FILE_PATH}"
  rm -f "${ENGINE_FILE_PATH}"
fi

# Fetch preprocessed_height, preprocessed_width and num_sensors from the config file
PREPROCESSED_HEIGHT=$(fetch_yaml_property_value "${CONFIG_FILE_PATH}" "preprocessed_height")
PREPROCESSED_WIDTH=$(fetch_yaml_property_value "${CONFIG_FILE_PATH}" "preprocessed_width")
NUM_SENSORS=$(fetch_yaml_property_value "${CONFIG_FILE_PATH}" "num_sensors")

# Update preprocessed dimensions if NETWORK_WIDTH and NETWORK_HEIGHT are set and not same as PREPROCESSED_WIDTH and PREPROCESSED_HEIGHT
if [[ "${NETWORK_WIDTH}" -ne 0 && "${NETWORK_HEIGHT}" -ne 0 && "${NETWORK_WIDTH}" -ne "${PREPROCESSED_WIDTH}" && "${NETWORK_HEIGHT}" -ne "${PREPROCESSED_HEIGHT}" ]]; then
  update_preprocessed_dims "${NETWORK_WIDTH}" "${NETWORK_HEIGHT}" "${CONFIG_FILE_PATH}" "${PREPROCESS_FILE_PATH}" "${NUM_SENSORS}"
  # Re-fetch the updated dimensions to use in trtexec
  PREPROCESSED_HEIGHT="${NETWORK_HEIGHT}"
  PREPROCESSED_WIDTH="${NETWORK_WIDTH}"
fi

# Extract model name from Sparse4D ONNX path to use in engine file name
echo "[INFO] Sparse4D ONNX model file: ${ONNX_MODEL_PATH}"
MODEL_NAME=$(basename "${ONNX_MODEL_PATH}" .onnx)
PRE_BUILT_ENGINE_FILE_PATH="/opt/nvidia/deepstream/deepstream/sources/sparse4d/${GPU_ARCH}-${MODEL_NAME}.engine"

# Check if the local engine file already exists
if [[ -f "${ENGINE_FILE_PATH}" ]]; then
  echo "[INFO] Using Saved TensorRT engine file for Sparse4D Model: ${ENGINE_FILE_PATH}"
# Check if the shared engine file already exists and FORCE_ENGINE_REBUILD is not set to True
elif [[ -f "${PRE_BUILT_ENGINE_FILE_PATH}" && "${FORCE_ENGINE_REBUILD}" != "True" ]]; then
  cp "${PRE_BUILT_ENGINE_FILE_PATH}" "${ENGINE_FILE_PATH}"
  echo "[INFO] Copying Pre-Built TensorRT engine file for Sparse4D Model: ${PRE_BUILT_ENGINE_FILE_PATH} to ${ENGINE_FILE_PATH}"
  echo "[INFO] Using Saved TensorRT engine file for Sparse4D Model: ${ENGINE_FILE_PATH}"
else
  # Use trtexec to generate the TensorRT engine from the ONNX model
  echo "[INFO] Generating TensorRT engine from Sparse4D ONNX model ${ONNX_MODEL_PATH}"
  echo "[INFO] MSDA plugin: ${MSDA_PLUGIN_PATH}"
  cd /opt/nvidia/deepstream/deepstream/sources/sparse4d/
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/sources/sparse4d/
  trtexec --onnx=$ONNX_MODEL_PATH \
          --fp16 \
          --saveEngine=$ENGINE_FILE_PATH \
          --builderOptimizationLevel=3 \
          --noDataTransfers \
          --noTF32 \
          --staticPlugins=$MSDA_PLUGIN_PATH \
          --useCudaGraph \
          --minShapes=img:1x1x3x${PREPROCESSED_HEIGHT}x${PREPROCESSED_WIDTH},projection_mat:1x1x4x4,image_wh:1x1x2,input_cached_anchor:1x600x11,input_cached_feature:1x600x256,prev_exists:1,interval_mask:1x1x1 \
          --maxShapes=img:1x${NUM_SENSORS}x3x${PREPROCESSED_HEIGHT}x${PREPROCESSED_WIDTH},projection_mat:1x${NUM_SENSORS}x4x4,image_wh:1x${NUM_SENSORS}x2,input_cached_anchor:1x600x11,input_cached_feature:1x600x256,prev_exists:1,interval_mask:1x1x1 \
          --optShapes=img:1x${NUM_SENSORS}x3x${PREPROCESSED_HEIGHT}x${PREPROCESSED_WIDTH},projection_mat:1x${NUM_SENSORS}x4x4,image_wh:1x${NUM_SENSORS}x2,input_cached_anchor:1x600x11,input_cached_feature:1x600x256,prev_exists:1,interval_mask:1x1x1

  # Save the Generated engine file
  cp "${ENGINE_FILE_PATH}" "${PRE_BUILT_ENGINE_FILE_PATH}"
  echo "[INFO] Saved the Generated TensorRT engine file for Sparse4D Model: ${PRE_BUILT_ENGINE_FILE_PATH}"
  echo "[INFO] Using Generated TensorRT engine file for Sparse4D Model: ${ENGINE_FILE_PATH}"
fi

OCR_INFERENCE=$(fetch_yaml_property_value "${CONFIG_FILE_PATH}" "generate_2d_bbox")
echo "[INFO] OCR Inference: ${OCR_INFERENCE}"

if [[ "${OCR_INFERENCE}" == "True" ]]; then
  #########################################################
  # Check if the shared OCR config file exists
  #########################################################
  SHARED_OCR_CONFIG_FILE_PATH="${SPARSE4D_CONFIG_PATH}config_ocr.yaml"
  OCR_CONFIG_FILE="/opt/nvidia/deepstream/deepstream/sources/sparse4d/configs/config_ocr.yaml"
  if [[ -f "${SHARED_OCR_CONFIG_FILE_PATH}" ]]; then
    # Check if the shared OCR config file is readable
    if [[ -r "${SHARED_OCR_CONFIG_FILE_PATH}" ]]; then
      # Copy OCR config file to destination directory
      cp "${SHARED_OCR_CONFIG_FILE_PATH}" "${OCR_CONFIG_FILE}"
      echo "[INFO] Copied Shared OCR Config file: ${SHARED_OCR_CONFIG_FILE_PATH} to ${OCR_CONFIG_FILE}"
    else
      echo "[WARN] Shared OCR Config file exists but is not accessible (permission issue): ${SHARED_OCR_CONFIG_FILE_PATH}"
    fi
  else
    echo "[WARN] Shared OCR Config file doesn't exist: ${SHARED_OCR_CONFIG_FILE_PATH}"
  fi
  # Validate OCR config file exists and is accessible
  validate_file_exists "OCR Config file" "${OCR_CONFIG_FILE}"

  #########################################################
  # Generate OCDNet engine
  #########################################################
  cd /opt/nvidia/deepstream/deepstream/sources/sparse4d/
  # Parse ocr config file
  if [[ -f "${OCR_CONFIG_FILE}" ]]; then
    # Extract nested YAML property: ocdnet.onnx_model_path
    OCD_MODEL_PATH=$(awk '/^ocdnet:/,0 {if (/^[[:space:]]+onnx_model_path:/) {sub(/^[[:space:]]+onnx_model_path:[[:space:]]*/, ""); gsub(/^"|"[[:space:]]*$|[[:space:]]*#.*$/, ""); print; exit}}' "${OCR_CONFIG_FILE}")
    MODEL_NAME=$(basename "${OCD_MODEL_PATH}" .onnx)
    # Validate OCDNet ONNX model file exists
    validate_file_exists "OCDNet ONNX model" "${OCD_MODEL_PATH}"
    # Extract nested YAML property: ocdnet.trt_engine_path
    ENGINE_FILE_PATH=$(awk '/^ocdnet:/,0 {if (/^[[:space:]]+trt_engine_path:/) {sub(/^[[:space:]]+trt_engine_path:[[:space:]]*/, ""); gsub(/^"|"[[:space:]]*$|[[:space:]]*#.*$/, ""); print; exit}}' "${OCR_CONFIG_FILE}")
    PRE_BUILT_ENGINE_FILE_PATH="/opt/nvidia/deepstream/deepstream/sources/sparse4d/${GPU_ARCH}-${MODEL_NAME}.engine"
    BATCH_SIZE=$(awk '/^ocdnet:/,0 {if (/^[[:space:]]+batch_size:/) {sub(/^[[:space:]]+batch_size:[[:space:]]*/, ""); gsub(/^"|"[[:space:]]*$|[[:space:]]*#.*$/, ""); print; exit}}' "${OCR_CONFIG_FILE}")
    # Extract infer_input_shape and parse into array [C, H, W]
    MODEL_DIMS_STR=$(awk '/^ocdnet:/,0 {if (/^[[:space:]]+infer_input_shape:/) {sub(/^[[:space:]]+infer_input_shape:[[:space:]]*/, ""); gsub(/^"|"[[:space:]]*$|[[:space:]]*#.*$/, ""); print; exit}}' "${OCR_CONFIG_FILE}")
    # Remove brackets and spaces, then split by comma into array
    MODEL_DIMS_STR=$(echo "${MODEL_DIMS_STR}" | tr -d '[]' | tr -d ' ')
    IFS=',' read -r -a MODEL_DIMS <<< "${MODEL_DIMS_STR}"
  else
    echo "[ERROR] OCR config file not found: ${OCR_CONFIG_FILE}" >&2
    exit 1
  fi
  # Engine file generation
  if [[ -f "${ENGINE_FILE_PATH}" ]]; then
    echo "[INFO] Using Saved TensorRT engine file for OCDNet Model: ${ENGINE_FILE_PATH}"
  elif [[ -f "${PRE_BUILT_ENGINE_FILE_PATH}" ]]; then
    cp "${PRE_BUILT_ENGINE_FILE_PATH}" "${ENGINE_FILE_PATH}" 
    echo "[INFO] Copying Pre-Built TensorRT engine file for OCDNet Model: ${PRE_BUILT_ENGINE_FILE_PATH} to ${ENGINE_FILE_PATH}"
    echo "[INFO] Using Saved TensorRT engine file for OCDNet Model: ${ENGINE_FILE_PATH}"
  else
    # Use trtexec to generate the TensorRT engine from the OCDNet model
    echo "[INFO] Generating TensorRT engine from OCDNet ONNX Model ${OCD_MODEL_PATH}"
    trtexec --onnx=${OCD_MODEL_PATH} \
            --minShapes=input:1x${MODEL_DIMS[0]}x${MODEL_DIMS[1]}x${MODEL_DIMS[2]} \
            --optShapes=input:${BATCH_SIZE}x${MODEL_DIMS[0]}x${MODEL_DIMS[1]}x${MODEL_DIMS[2]} \
            --maxShapes=input:${BATCH_SIZE}x${MODEL_DIMS[0]}x${MODEL_DIMS[1]}x${MODEL_DIMS[2]} \
            --fp16 \
            --saveEngine=${ENGINE_FILE_PATH}
    # Save the Generated engine file
    cp "${ENGINE_FILE_PATH}" "${PRE_BUILT_ENGINE_FILE_PATH}"
    echo "[INFO] Saved the generated TensorRT engine file for OCDNet Model: ${PRE_BUILT_ENGINE_FILE_PATH}"
    echo "[INFO] Using Generated TensorRT engine file for OCDNet Model: ${ENGINE_FILE_PATH}"
  fi

  #########################################################
  # Generate OCRNet engine
  #########################################################
  # Extract onnx model path for OCRNet
  OCR_MODEL_PATH=$(awk '/^ocrnet:/,0 {if (/^[[:space:]]+onnx_model_path:/) {sub(/^[[:space:]]+onnx_model_path:[[:space:]]*/, ""); gsub(/^"|"[[:space:]]*$|[[:space:]]*#.*$/, ""); print; exit}}' "${OCR_CONFIG_FILE}")
  MODEL_NAME=$(basename "${OCR_MODEL_PATH}" .onnx)
  # Validate OCRNet ONNX model file exists
  validate_file_exists "OCRNet ONNX model" "${OCR_MODEL_PATH}"
  # Extract engine file path for OCRNet
  ENGINE_FILE_PATH=$(awk '/^ocrnet:/,0 {if (/^[[:space:]]+trt_engine_path:/) {sub(/^[[:space:]]+trt_engine_path:[[:space:]]*/, ""); gsub(/^"|"[[:space:]]*$|[[:space:]]*#.*$/, ""); print; exit}}' "${OCR_CONFIG_FILE}")
  PRE_BUILT_ENGINE_FILE_PATH="/opt/nvidia/deepstream/deepstream/sources/sparse4d/${GPU_ARCH}-${MODEL_NAME}.engine"
  # Extract batch_size for OCRNet
  BATCH_SIZE=$(awk '/^ocrnet:/,0 {if (/^[[:space:]]+batch_size:/) {sub(/^[[:space:]]+batch_size:[[:space:]]*/, ""); gsub(/^"|"[[:space:]]*$|[[:space:]]*#.*$/, ""); print; exit}}' "${OCR_CONFIG_FILE}")
  # Extract dict_file for OCRNet
  DICT_FILE=$(awk '/^ocrnet:/,0 {if (/^[[:space:]]+dict_file:/) {sub(/^[[:space:]]+dict_file:[[:space:]]*/, ""); gsub(/^"|"[[:space:]]*$|[[:space:]]*#.*$/, ""); print; exit}}' "${OCR_CONFIG_FILE}")
  # Validate dictionary file exists and is readable
  validate_file_exists "OCRNet dictionary file" "${DICT_FILE}"
  # Extract infer_input_shape and parse into array [C, H, W]
  MODEL_DIMS_STR=$(awk '/^ocrnet:/,0 {if (/^[[:space:]]+infer_input_shape:/) {sub(/^[[:space:]]+infer_input_shape:[[:space:]]*/, ""); gsub(/^"|"[[:space:]]*$|[[:space:]]*#.*$/, ""); print; exit}}' "${OCR_CONFIG_FILE}")
  # Remove brackets and spaces, then split by comma into array
  MODEL_DIMS_STR=$(echo "${MODEL_DIMS_STR}" | tr -d '[]' | tr -d ' ')
  IFS=',' read -r -a MODEL_DIMS <<< "${MODEL_DIMS_STR}"
  # Engine file generation
  if [[ -f "${ENGINE_FILE_PATH}" ]]; then
    echo "[INFO] Using Saved TensorRT engine file for OCRNet Model: ${ENGINE_FILE_PATH}"
  elif [[ -f "${PRE_BUILT_ENGINE_FILE_PATH}" ]]; then
    cp "${PRE_BUILT_ENGINE_FILE_PATH}" "${ENGINE_FILE_PATH}" 
    echo "[INFO] Copying Pre-Built TensorRT engine file for OCRNet Model: ${PRE_BUILT_ENGINE_FILE_PATH} to ${ENGINE_FILE_PATH}"
    echo "[INFO] Using Saved TensorRT engine file for OCRNet Model: ${ENGINE_FILE_PATH}"
  elif [[ -f "${ENGINE_FILE_PATH}" ]]; then
    echo "[INFO] Using Saved TensorRT engine file for OCRNet Model: ${ENGINE_FILE_PATH}"
  else
    # Use trtexec to generate the TensorRT engine from the OCRNet model
    echo "[INFO] Generating TensorRT engine from OCRNet ONNX Model ${OCR_MODEL_PATH}"
    trtexec --onnx=${OCR_MODEL_PATH} \
            --minShapes=input:1x${MODEL_DIMS[0]}x${MODEL_DIMS[1]}x${MODEL_DIMS[2]} \
            --optShapes=input:${BATCH_SIZE}x${MODEL_DIMS[0]}x${MODEL_DIMS[1]}x${MODEL_DIMS[2]} \
            --maxShapes=input:${BATCH_SIZE}x${MODEL_DIMS[0]}x${MODEL_DIMS[1]}x${MODEL_DIMS[2]} \
            --fp16 \
            --saveEngine=${ENGINE_FILE_PATH}
    # Save the Generated engine file
    cp "${ENGINE_FILE_PATH}" "${PRE_BUILT_ENGINE_FILE_PATH}"
    echo "[INFO] Saved the generated TensorRT engine file for OCRNet Model: ${PRE_BUILT_ENGINE_FILE_PATH}"
    echo "[INFO] Using Generated TensorRT engine file for OCRNet Model: ${ENGINE_FILE_PATH}"
  fi
fi

echo "[SUCCESS] Sparse4D model setup complete"
