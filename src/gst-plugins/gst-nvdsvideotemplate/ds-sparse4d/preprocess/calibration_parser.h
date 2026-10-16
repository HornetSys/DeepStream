/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CAMERA_CALIBRATION_PARSER_H
#define CAMERA_CALIBRATION_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <nlohmann/json.hpp> // JSON parsing library
#include <Eigen/Dense>       // For matrix operations
#include <variant>
#include <iostream>
#include <fstream>
#include <regex>

using json = nlohmann::json;
using namespace Eigen;

// Type definitions
using Matrix4f = Eigen::Matrix<float, 4, 4>;
using Matrix3f = Eigen::Matrix<float, 3, 3>;
using Matrix4d = Eigen::Matrix<double, 4, 4>;
using Matrix3d = Eigen::Matrix<double, 3, 3>;
using Vector3d = Eigen::Matrix<double, 3, 1>;
using Vector3f = Eigen::Vector3f;
using MatrixXd = Eigen::MatrixXd;
using AugConfig = std::map<std::string, std::variant<float, std::vector<int>, bool>>;

// Constants
const float FPS = 30.0;

// Function declarations

// Helper function to convert an Eigen matrix to JSON
template <typename Derived>
json eigenMatrixToJson(const Eigen::MatrixBase<Derived> &matrix);

// Extract all numbers from a string
std::vector<int> extractNumbers(const std::string& text);

// Extract camera name from a string based on numerical values
std::string getCamNameFromString(const std::string& str, int numIndex = -1);

// Load camera calibration information for BEVFormer scene from a JSON file
json loadCalibBEVFormerScene(const std::string& calibFilePath);

// Load camera calibration data from a synthetic JSON file and convert it to a calibration dictionary
json loadCalibSyntheticJson(const std::string& calibFilePath, bool recentering = false);

// Loads group information from a dictionary if the type matches the provided key
std::pair<std::string, std::map<std::string, std::vector<float>>> loadGroup(
    const json& groupDict, const std::string& key);

// Load calibration information for synthetic camera groups from a JSON file
std::pair<std::map<std::string, json>, std::map<std::string, json>>
loadCalibSyntheticCameraGroupsJson(const std::string &calibFilePath, bool recentering = false);

// Load camera calibration data from file (supports different modes and grouping)
std::pair<json, json> loadCalib(const std::string& calibFilePath, const std::string& calibMode = "synthetic",
                               bool useCameraGroups = true, bool recentering = false);

// Function to normalize matrices (replace -0.0 with +0.0)
void normalizeMatrix(Matrix4d& matrix);

// Read and parse camera calibration information from a calibration file
json fillCameraInfos(const std::string& calibFilePath, const std::string& bevGroupName,
                    const std::string& calibMode = "synthetic", bool useCameraGroups = true,
                    bool recentering = false);

// Compute camera calibration matrices from camera information
std::vector<Matrix4f> getCameraCalibrationMatrix(const json& info);

// Parse calibration file and return lidar-to-image transformation matrices, camera keys and group origin
std::tuple<std::vector<Matrix4f>, std::vector<std::string>, std::vector<float>> parseCalibFile(
    const std::string& calibFilePath, const std::string& bevGroupName,
    const std::string& calibMode = "synthetic", bool useCameraGroups = true,
    bool recentering = false);

// Fetch scene map information (scale factor and translation to global coordinates)
std::pair<float, std::vector<float>> fetchSceneMapInfo(const std::string& calibFilePath);

// Helper function to compute object velocities based on positions in consecutive frames
std::vector<std::vector<float>> getObjectVelocity(
    const std::vector<std::vector<float>>& locations,
    const std::vector<int>& instanceIds,
    const json& annotationsPrev);

// Fills camera information with ground truth data for BEV scene analysis
std::vector<json> fillCameraInfosGT(
    const std::string& calibFilePath, const std::string& bevGroupName,
    const std::string& groundTruthFilePath, const std::string& sceneName,
    const std::vector<std::string>& classNames, int startFrame = 0, int frames = 1000);

// Transform lidar information based on augmentation configurations
std::vector<Matrix4f> transformLidarInfo(
    const std::vector<Matrix4f> &lidar2img,
    AugConfig &augConfigs, int height = 1080, int width = 1920);

#endif // CAMERA_CALIBRATION_PARSER_H
