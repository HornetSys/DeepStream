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

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <regex>
#include <variant>

#include "calibration_parser.h"

#include <nlohmann/json.hpp>
#include <Eigen/Dense>

using json = nlohmann::json;
using namespace Eigen;

using Matrix4f = Eigen::Matrix<float, 4, 4>;
using Matrix3f = Eigen::Matrix<float, 3, 3>;
using Matrix4d = Eigen::Matrix<double, 4, 4>;
using Matrix3d = Eigen::Matrix<double, 3, 3>;
using Vector3d = Eigen::Matrix<double, 3, 1>;
using Vector3f = Eigen::Vector3f;
using MatrixXd = Eigen::MatrixXd;

// Constants

// Helper function to convert an Eigen matrix to JSON
template <typename Derived>
json eigenMatrixToJson(const Eigen::MatrixBase<Derived> &matrix)
{
    json j = json::array();
    for (int i = 0; i < matrix.rows(); ++i)
    {
        json row = json::array();
        for (int j = 0; j < matrix.cols(); ++j)
        {
            row.push_back(matrix(i, j));
        }
        j.push_back(row);
    }
    return j;
}

// Extract all numbers from a string using regular expressions
std::vector<int> extractNumbers(const std::string &text)
{
    std::regex re("\\d+"); // Regular expression to match sequences of digits
    std::sregex_iterator words_begin(text.begin(), text.end(), re);
    std::sregex_iterator words_end;

    std::vector<int> numbers;
    for (std::sregex_iterator i = words_begin; i != words_end; ++i)
    {
        std::smatch match = *i; // Extract the matched sequence
        numbers.push_back(std::stoi(match.str())); // Convert the matched string to an integer
    }
    return numbers;
}

// Extract camera name from a string based on numerical values
std::string getCamNameFromString(const std::string &str, int numIndex)
{
    // Extract all numbers from the string
    std::vector<int> numbers = extractNumbers(str);

    if (numbers.empty())
    {
        return "";
    }

    if (numIndex == -1)
    {
        // Return the last number in the string
        return std::to_string(numbers.back());
    }
    else if (numIndex >= 0 && static_cast<std::size_t>(numIndex) < numbers.size())
    {
        // Safe comparison: cast numIndex to std::size_t
        return std::to_string(numbers[numIndex]);
    }
    else
    {
        return "";
    }
}

// Function to load group information from JSON
std::pair<std::string, std::map<std::string, std::vector<float>>> loadGroup(
    const json &groupDict, const std::string &key) {
    try {
        // Check if required fields exist
        if (!groupDict.contains("type") || !groupDict.contains("name")) {
            return {"", {}};
        }

        // First check if the type matches the key
        if (groupDict["type"].get<std::string>() != key) {
            return {"", {}};
        }

        // If we get here, the type matches the key, so extract the group name
        std::string groupName = groupDict["name"].get<std::string>();
        std::map<std::string, std::vector<float>> groupInfoDict;

        // Safely extract origin - handle various possible formats
        if (groupDict.contains("origin")) {
            std::vector<float> origin;

            if (groupDict["origin"].is_array()) {
                // Handle array format directly
                origin = groupDict["origin"].get<std::vector<float>>();
            }
            else if (groupDict["origin"].is_object()) {
                // Handle x, y coordinates in object
                if (groupDict["origin"].contains("x") && groupDict["origin"].contains("y")) {
                    origin.push_back(groupDict["origin"]["x"].get<float>());
                    origin.push_back(groupDict["origin"]["y"].get<float>());
                }
            }
            else {
                origin = {0.0f, 0.0f}; // Default values
            }

            groupInfoDict["origin"] = origin;
        }

        // Safely extract dimensions - handle various possible formats
        if (groupDict.contains("dimensions")) {
            std::vector<float> dimensions;

            if (groupDict["dimensions"].is_array()) {
                // Handle array format directly
                dimensions = groupDict["dimensions"].get<std::vector<float>>();
            } 
            else if (groupDict["dimensions"].is_object()) {
                // Extract common dimension fields
                if (groupDict["dimensions"].contains("width")) {
                    dimensions.push_back(groupDict["dimensions"]["width"].get<float>());
                }
                if (groupDict["dimensions"].contains("height")) {
                    dimensions.push_back(groupDict["dimensions"]["height"].get<float>());
                }
                if (groupDict["dimensions"].contains("x")) {
                    dimensions.push_back(groupDict["dimensions"]["x"].get<float>());
                }
                if (groupDict["dimensions"].contains("y")) {
                    dimensions.push_back(groupDict["dimensions"]["y"].get<float>());
                }
            }

            groupInfoDict["dimensions"] = dimensions;
        }

        return {groupName, groupInfoDict};
    }
    catch (const std::exception &e) {
        std::cerr << "Exception in loadGroup: " << e.what() << std::endl;
        return {"", {}};
    }
}

// Load camera calibration information from a BEVFormer scene JSON file
json loadCalibBEVFormerScene(const std::string &calibFilePath)
{
    try
    {
        std::ifstream file(calibFilePath);

        if (!file.is_open())
        {
            std::cerr << "Error: Failed to open file " << calibFilePath << '\n';
            return {};
        }

        json calibration;
        file >> calibration;

        return calibration;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return {};
    }
}

// Load camera calibration data from a synthetic JSON file and convert it to a calibration dictionary
json loadCalibSyntheticJson(const std::string &calibFilePath, bool recentering)
{
    try
    {
        std::ifstream file(calibFilePath);

        if (!file.is_open())
        {
            std::cerr << "Error: Failed to open file " << calibFilePath << '\n';
            return {};
        }

        json calibJson;
        file >> calibJson;

        json calibDict;

        for (const auto &sensor : calibJson["sensors"])
        {
            std::string camName = sensor["id"];

            Matrix4f intrinExt = Matrix4f::Identity();
            intrinExt.topLeftCorner<3, 3>() =
                Eigen::Map<const Matrix3f>(sensor["intrinsicMatrix"].get<std::vector<float>>().data());

            Matrix4f extrin = Matrix4f::Identity();
            extrin.topLeftCorner<3, 4>() =
                Eigen::Map<const Eigen::Matrix<float, 3, 4>>(sensor["extrinsicMatrix"].get<std::vector<float>>().data());

            Matrix4f projW2C;

            if (recentering) {
                projW2C = extrin;
            } else {
                // Left-handed to right-handed conversion
                extrin.col(1) = -extrin.col(1);
                extrin.col(2) = -extrin.col(2);
                projW2C = extrin.inverse();
            }

            Matrix4f projW2P = intrinExt * projW2C;

            calibDict[camName]["intrinsic matrix"] = eigenMatrixToJson(intrinExt.topLeftCorner<3, 3>());
            calibDict[camName]["projection matrix w2c"] = eigenMatrixToJson(projW2C);
            calibDict[camName]["projection matrix w2p"] = eigenMatrixToJson(projW2P);
        }

        return calibDict;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return {};
    }
}

#include <cmath> // For std::fabs

// Function to replace -0.0 with +0.0 in a matrix
void normalizeMatrix(Matrix4d& matrix) {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            if (std::fabs(matrix(i, j)) == 0.0) {
                matrix(i, j) = 0.0; // Replace -0.0 with +0.0
            }
        }
    }
}

// Function to load calibration data for synthetic camera groups
std::pair<std::map<std::string, json>, std::map<std::string, json>>
loadCalibSyntheticCameraGroupsJson(const std::string &calibFilePath, bool recentering) {
    try {
        // Open the JSON file
        std::ifstream file(calibFilePath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + calibFilePath);
        }

        // Parse the JSON file
        json calibJson;
        file >> calibJson;

        // Initialize dictionaries to store calibration and group area data
        std::map<std::string, json> calibDict;
        std::map<std::string, json> groupAreaDict;

        // Process each sensor in the JSON
        for (const auto &sensor : calibJson["sensors"]) {
            std::string camName = sensor["id"];

            // Get BEV group information
            if (!sensor.contains("group")) {
                std::cerr << "Sensor " << camName << " does not have a 'group' field" << std::endl;
                continue;
            }

            auto [bevGroupName, bevGroupInfoDict] = loadGroup(sensor["group"], "bev");

            if (bevGroupName.empty()) {
                continue;
            }

            // Initialize the dictionaries for this BEV group if not already done
            if (calibDict.find(bevGroupName) == calibDict.end()) {
                calibDict[bevGroupName] = json::object();

                // Convert std::map to json for group area info
                json groupAreaInfo = json::object();
                for (const auto& [key, value] : bevGroupInfoDict) {
                    groupAreaInfo[key] = value;
                }
                groupAreaDict[bevGroupName] = groupAreaInfo;
            }

            try {
                // Create intrinsic matrix with safer conversion
                Matrix4f intrinExt = Matrix4f::Identity();
                Matrix3f intrinsicMatrix = Matrix3f::Zero();

                // Check if intrinsicMatrix exists
                if (!sensor.contains("intrinsicMatrix")) {
                    std::cerr << "Sensor " << camName << " does not have intrinsicMatrix" << std::endl;
                    continue;
                }

                // Check intrinsicMatrix format and handle accordingly
                if (sensor["intrinsicMatrix"].is_array()) {
                    if (sensor["intrinsicMatrix"].size() == 9) {
                        // Flat array format
                        std::vector<float> intrinsicFlat = sensor["intrinsicMatrix"].get<std::vector<float>>();
                        for (int i = 0; i < 3; i++) {
                            for (int j = 0; j < 3; j++) {
                                intrinsicMatrix(i, j) = intrinsicFlat[i * 3 + j];
                            }
                        }
                    } else if (sensor["intrinsicMatrix"].size() == 3 &&
                              sensor["intrinsicMatrix"][0].is_array() &&
                              sensor["intrinsicMatrix"][0].size() == 3) {
                        // Nested array format
                        for (int i = 0; i < 3; i++) {
                            for (int j = 0; j < 3; j++) {
                                intrinsicMatrix(i, j) = sensor["intrinsicMatrix"][i][j].get<float>();
                            }
                        }
                    } else {
                        std::cerr << "Unexpected intrinsicMatrix format for sensor " << camName << std::endl;
                        continue;
                    }
                } else {
                    std::cerr << "intrinsicMatrix is not an array for sensor " << camName << std::endl;
                    continue;
                }

                intrinExt.topLeftCorner<3, 3>() = intrinsicMatrix;

                // Check if extrinsicMatrix exists
                if (!sensor.contains("extrinsicMatrix")) {
                    std::cerr << "Sensor " << camName << " does not have extrinsicMatrix" << std::endl;
                    continue;
                }

                // Create extrinsic matrix with safer conversion
                Matrix4f extrin = Matrix4f::Identity();
                Matrix<float, 3, 4> extrinsicMatrix = Matrix<float, 3, 4>::Zero();

                // Check extrinsicMatrix format and handle accordingly
                if (sensor["extrinsicMatrix"].is_array()) {
                    if (sensor["extrinsicMatrix"].size() == 12) {
                        // Flat array format
                        std::vector<float> extrinsicFlat = sensor["extrinsicMatrix"].get<std::vector<float>>();
                        for (int i = 0; i < 3; i++) {
                            for (int j = 0; j < 4; j++) {
                                extrinsicMatrix(i, j) = extrinsicFlat[i * 4 + j];
                            }
                        }
                    } else if (sensor["extrinsicMatrix"].size() == 3 &&
                              sensor["extrinsicMatrix"][0].is_array() &&
                              sensor["extrinsicMatrix"][0].size() == 4) {
                        // Nested array format
                        for (int i = 0; i < 3; i++) {
                            for (int j = 0; j < 4; j++) {
                                extrinsicMatrix(i, j) = sensor["extrinsicMatrix"][i][j].get<float>();
                            }
                        }
                    } else {
                        std::cerr << "Unexpected extrinsicMatrix format for sensor " << camName << std::endl;
                        continue;
                    }
                } else {
                    std::cerr << "extrinsicMatrix is not an array for sensor " << camName << std::endl;
                    continue;
                }

                extrin.topLeftCorner<3, 4>() = extrinsicMatrix;

                // Use extrinsic matrix directly as world-to-camera projection
                Matrix4f projW2C = extrin;

                if (recentering && !groupAreaDict.empty()) {
                    // Convert to camera-to-world, apply translation, then back to world-to-camera
                    Matrix4f projC2W = projW2C.inverse();

                    // Subtract group origin from camera position
                    std::vector<float> origin = groupAreaDict[bevGroupName]["origin"];
                    projC2W(0, 3) -= origin[0];  // X coordinate
                    projC2W(1, 3) -= origin[1];  // Y coordinate

                    // Convert back to world-to-camera
                    projW2C = projC2W.inverse();
                }

                // Calculate world-to-pixel projection matrix
                Matrix4f projW2P = intrinExt * projW2C;

                // Store the matrices in the calibration dictionary
                json calibDictCam;
                calibDictCam["intrinsic matrix"] = eigenMatrixToJson(intrinExt.topLeftCorner<3, 3>());
                calibDictCam["projection matrix w2c"] = eigenMatrixToJson(projW2C);
                calibDictCam["projection matrix w2p"] = eigenMatrixToJson(projW2P);

                calibDict[bevGroupName][camName] = calibDictCam;
            } catch (const std::exception &e) {
                std::cerr << "Error processing matrices for sensor " << camName << ": " << e.what() << std::endl;
                continue;  // Skip this sensor and try the next one
            }
        }

        return {calibDict, groupAreaDict};
    } catch (const std::exception &e) {
        std::cerr << "Error loading calibration data: " << e.what() << std::endl;
        return {{}, {}};
    }
}

// Load camera calibration data from file
std::pair<json, json> loadCalib(const std::string &calibFilePath, const std::string &calibMode, bool useCameraGroups, bool recentering) {
    json groupAreaDict = nullptr;
    json calibDict = json::object(); // Initialize as empty object

    try {
        if (calibMode == "synthetic") {
            if (useCameraGroups) {
                auto [calib, groupArea] = loadCalibSyntheticCameraGroupsJson(calibFilePath, recentering);

                // Convert from map to json
                for (const auto &[groupName, groupData] : calib) {
                    calibDict[groupName] = groupData;
                }

                // Convert from map to json
                groupAreaDict = json::object();
                for (const auto &[groupName, groupData] : groupArea) {
                    groupAreaDict[groupName] = groupData;
                }
            } else {
                calibDict = loadCalibSyntheticJson(calibFilePath, recentering);
            }
        } else {
            calibDict = loadCalibBEVFormerScene(calibFilePath);
        }

        if (calibDict.empty()) {
            std::cerr << "Warning: Calibration dictionary is empty after loading " << calibFilePath << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in loadCalib: " << e.what() << std::endl;
    }

    return {calibDict, groupAreaDict};
}

// Read and parse camera calibration information from a calibration file
json fillCameraInfos(const std::string &calibFilePath, const std::string &bevGroupName,
                     const std::string &calibMode, bool useCameraGroups, bool recentering) {
    // Load the calib file, grouped on the basis bev sensor name
    auto [calibDict, groupAreaDict] = loadCalib(calibFilePath, calibMode, useCameraGroups, recentering);

    // Verify Bev-Group Name is present in the calib file
    if (calibDict.empty()) {
        std::cerr << "Error: Calibration dictionary is empty after loading " << calibFilePath << std::endl;
        throw std::runtime_error("Empty calibration dictionary");
    }

    // Check if the requested BEV group exists
    if (!calibDict.contains(bevGroupName)) {
        std::cerr << "Error: No BEV Sensor Name '" << bevGroupName << "' found in " << calibFilePath << std::endl;
        std::cerr << "Available groups: ";
        for (const auto& [group, _] : calibDict.items()) {
            std::cerr << "'" << group << "' ";
        }
        std::cerr << std::endl;

        // As a fallback, try using the first available group and warn the user
        if (!calibDict.empty()) {
            std::string firstGroup = calibDict.begin().key();
            std::cerr << "WARNING: Using first available group '" << firstGroup << "' as fallback" << std::endl;

            json info = {{"cams", json::object()}};
            info["bev_group_name"] = firstGroup;
            info["group_origin"] = groupAreaDict[firstGroup]["origin"];

            for (const auto &[cam, camData] : calibDict[firstGroup].items()) {
                json camInfo = json::object();

                // Convert JSON arrays to Eigen matrices for internal use
                Matrix3f camIntrinsic = Matrix3f::Zero();
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        camIntrinsic(i, j) = camData["intrinsic matrix"][i][j];
                    }
                }

                Matrix4f camSensor2World = Matrix4f::Zero();
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        camSensor2World(i, j) = camData["projection matrix w2c"][i][j];
                    }
                }

                camInfo["cam_intrinsic"] = eigenMatrixToJson(camIntrinsic);
                camInfo["sensor2world_transform"] = eigenMatrixToJson(camSensor2World);
                info["cams"][cam] = camInfo;
            }

            return info;
        }

        throw std::runtime_error("BEV group name not found in calibration file");
    }

    json info = {{"cams", json::object()}};
    info["bev_group_name"] = bevGroupName;
    info["group_origin"] = groupAreaDict[bevGroupName]["origin"];

    for (const auto &[cam, camData] : calibDict[bevGroupName].items()) {
        json camInfo = json::object();

        // Convert JSON arrays to Eigen matrices for internal use
        Matrix3f camIntrinsic = Matrix3f::Zero();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                camIntrinsic(i, j) = camData["intrinsic matrix"][i][j];
            }
        }

        Matrix4f camSensor2World = Matrix4f::Zero();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                camSensor2World(i, j) = camData["projection matrix w2c"][i][j];
            }
        }
        
        camInfo["cam_intrinsic"] = eigenMatrixToJson(camIntrinsic);
        camInfo["sensor2world_transform"] = eigenMatrixToJson(camSensor2World);
        info["cams"][cam] = camInfo;
    }

    return info;
}

std::vector<Matrix4f> getCameraCalibrationMatrix(const json &info) {
    std::vector<Matrix4f> lidar2imgRts;
    std::vector<std::string> sensorNames;

    for (const auto &[camType, camInfo] : info["cams"].items()) {
        sensorNames.push_back(camType);

        // Extract and convert sensor2world transform matrix (extrinsic)
        Matrix4f cam2worldTransform = Matrix4f::Identity();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cam2worldTransform(i, j) = camInfo["sensor2world_transform"][i][j];
            }
        }

        // Extract and convert camera intrinsic matrix
        Matrix3f intrinsic = Matrix3f::Identity();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                intrinsic(i, j) = camInfo["cam_intrinsic"][i][j];
            }
        }

        // Create viewpad (extended intrinsic matrix)
        Matrix4f viewpad = Matrix4f::Identity();
        viewpad.block<3, 3>(0, 0) = intrinsic;

        // Compute the projection matrix
        Matrix4f lidar2imgRt = viewpad * cam2worldTransform;
        lidar2imgRts.push_back(lidar2imgRt);
    }

    return lidar2imgRts;
}

// Parse calibration file and return lidar-to-image transformation matrices, camera keys and group origin
std::tuple<std::vector<Matrix4f>, std::vector<std::string>, std::vector<float>> parseCalibFile(
    const std::string &calibFilePath,
    const std::string &bevGroupName,
    const std::string &calibMode,
    bool useCameraGroups,
    bool recentering) {

    try {
        json dataInfo = fillCameraInfos(calibFilePath, bevGroupName, calibMode, useCameraGroups, recentering);

        std::vector<Matrix4f> lidar2imgTensor = getCameraCalibrationMatrix(dataInfo);

        // Extract camera keys
        std::vector<std::string> cameraKeys;
        for (const auto &[cam, _] : dataInfo["cams"].items()) {
            cameraKeys.push_back(cam);
        }
        std::vector<float> groupOrigin = dataInfo["group_origin"].get<std::vector<float>>();

        return std::make_tuple(lidar2imgTensor, cameraKeys, groupOrigin);
    } catch (const std::exception &e) {
        std::cerr << "Error in parseCalibFile: " << e.what() << std::endl;

        // Return empty results on error
        return std::make_tuple(std::vector<Matrix4f>{}, std::vector<std::string>{}, std::vector<float>{});
    }
}

std::pair<float, std::vector<float>> fetchSceneMapInfo(const std::string &calibFilePath) {
    try {
        std::ifstream file(calibFilePath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open calibration file: " + calibFilePath);
        }

        json calibDict;
        file >> calibDict;

        // Get the first sensor's information
        if (calibDict.contains("sensors") && !calibDict["sensors"].empty()) {
            const auto &sensor = calibDict["sensors"][0];

            if (sensor.contains("scaleFactor") && sensor.contains("translationToGlobalCoordinates")) {
                float scaleFactor = sensor["scaleFactor"];
                std::vector<float> translationToGlobalCoordinates = sensor["translationToGlobalCoordinates"];
                return {scaleFactor, translationToGlobalCoordinates};
            }
        }

        throw std::runtime_error("Required calibration data not found in file");
    } catch (const std::exception &e) {
        std::cerr << "Error fetching scene map info: " << e.what() << std::endl;
        return {1.0f, {0.0f, 0.0f, 0.0f}}; // Default values
    }
}

std::vector<std::vector<float>> getObjectVelocity(
    const std::vector<std::vector<float>> &locations,
    const std::vector<int> &instanceIds,
    const json &annotationsPrev) {

    size_t numObjects = locations.size();
    std::vector<std::vector<float>> velocities(numObjects, std::vector<float>(2, 0.0f));

    if (annotationsPrev.empty()) {
        return velocities; // Return zero velocities if no previous annotations
    }

    // Copy current locations as default (will result in zero velocity if object not found in prev frame)
    std::vector<std::vector<float>> locationsPrev = locations;

    // Create mapping from object ID to index
    std::map<int, size_t> objIdToIdx;
    for (size_t i = 0; i < instanceIds.size(); ++i) {
        objIdToIdx[instanceIds[i]] = i;
    }

    // Update locations from previous frame for objects that exist in both frames
    for (const auto &anno : annotationsPrev) {
        int objId = anno["object id"];
        if (objIdToIdx.find(objId) != objIdToIdx.end()) {
            // Object found in current frame, update its previous location
            locationsPrev[objIdToIdx[objId]] = anno["3d location"].get<std::vector<float>>();
        }
    }

    // Calculate velocities (distance * FPS)
    for (size_t i = 0; i < numObjects; ++i) {
        std::vector<float> velocity(2);
        for (int j = 0; j < 2; ++j) { // Only keep vx, vy
            velocity[j] = (locations[i][j] - locationsPrev[i][j]) * FPS;
        }

        // Check if velocity is too high (handling frame drop bug)
        float velocityNorm = std::sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1]);
        if (velocityNorm > 5.0f) {
            velocity[0] = std::numeric_limits<float>::quiet_NaN();
            velocity[1] = std::numeric_limits<float>::quiet_NaN();
        }

        velocities[i] = velocity;
    }

    return velocities;
}

std::vector<json> fillCameraInfosGT(
    const std::string &calibFilePath,
    const std::string &bevGroupName,
    const std::string &groundTruthFilePath,
    const std::string &sceneName,
    const std::vector<std::string>& classNames,
    int startFrame,
    int frames) {

    bool recentering = false;
    std::vector<json> infos;

    // Load the calibration file
    auto [calibDict, groupAreaDict] = loadCalib(calibFilePath);

    // Load the ground truth file
    std::ifstream gtFile(groundTruthFilePath);
    if (!gtFile.is_open()) {
        throw std::runtime_error("Failed to open ground truth file: " + groundTruthFilePath);
    }

    json groundTruthsDict;
    gtFile >> groundTruthsDict;

    // Verify Bev-Group Name is present in the calib file
    if (!calibDict.contains(bevGroupName)) {
        throw std::runtime_error("No BEV Sensor Name " + bevGroupName + " found in " + calibFilePath);
    }

    std::string subSceneName = sceneName + "+" + bevGroupName;

    for (int frameId = startFrame; frameId < startFrame + frames; ++frameId) {
        json info = {
            {"frame_idx", frameId},
            {"cams", json::object()},
            {"scene_name", sceneName},
            {"timestamp", frameId / FPS},
            {"token", subSceneName + "_" + std::to_string(frameId)},
            {"scene_name", subSceneName},
            {"bev_group_name", bevGroupName}
        };

        std::vector<float> groupOrigin = {0.0f, 0.0f};
        if (recentering && !groupAreaDict.is_null() && groupAreaDict.contains(bevGroupName)) {
            groupOrigin = groupAreaDict[bevGroupName]["origin"].get<std::vector<float>>();
        }

        // Process camera information
        for (const auto &[cam, camData] : calibDict[bevGroupName].items()) {
            std::string sdToken = info["token"].get<std::string>() + "+" + cam;

            Matrix3f camIntrinsic = Matrix3f::Zero();
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    camIntrinsic(i, j) = camData["intrinsic matrix"][i][j];
                }
            }

            Matrix4f camSensor2World = Matrix4f::Zero();
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    camSensor2World(i, j) = camData["projection matrix w2c"][i][j];
                }
            }

            if (recentering && !groupAreaDict.is_null()) {
                // Shift camera to group center
                Matrix4f camWorld2Sensor = camSensor2World.inverse();
                camWorld2Sensor(0, 3) -= groupOrigin[0];
                camWorld2Sensor(1, 3) -= groupOrigin[1];
                camSensor2World = camWorld2Sensor.inverse();
            }

            json camInfo = {
                {"sample_data_token", sdToken},
                {"cam_intrinsic", eigenMatrixToJson(camIntrinsic)},
                {"sensor2world_transform", eigenMatrixToJson(camSensor2World)}
            };

            info["cams"][cam] = camInfo;
        }

        // Get annotations for evaluation
        json annotations, annotationsPrev;
        std::string frameIdStr = std::to_string(frameId);
        std::string prevFrameIdStr = std::to_string(frameId - 1);

        if (groundTruthsDict.contains(frameIdStr)) {
            annotations = groundTruthsDict[frameIdStr];
        }

        if (groundTruthsDict.contains(prevFrameIdStr)) {
            annotationsPrev = groundTruthsDict[prevFrameIdStr];
        }

        // Process annotations
        if (!annotations.empty()) {
            std::vector<std::vector<float>> locs;
            std::vector<std::vector<float>> dims;
            std::vector<float> rots;
            std::vector<int> instanceIds;
            std::vector<std::string> names;
            std::vector<bool> validFlag;
            std::vector<json> boxes2dFull;
            std::vector<json> boxes2dVisible;

            for (const auto &anno : annotations) {
                // Fix JSON to vector conversions
                locs.push_back(anno["3d location"].get<std::vector<float>>());
                dims.push_back(anno["3d bounding box scale"].get<std::vector<float>>());
                rots.push_back(anno["3d bounding box rotation"][2]);
                instanceIds.push_back(anno["object id"]);
                names.push_back(anno["object type"]);
                validFlag.push_back(true);
                boxes2dFull.push_back(anno["2d bounding box"]);
                boxes2dVisible.push_back(anno["2d bounding box visible"]);
            }

            // Calculate velocities
            std::vector<std::vector<float>> velocity = getObjectVelocity(locs, instanceIds, annotationsPrev);

            // Adjust height to center of 3D box
            for (size_t i = 0; i < locs.size(); ++i) {
                locs[i][2] += dims[i][2] / 2.0f;
            }

            // Apply recentering if needed
            if (recentering && !groupAreaDict.is_null()) {
                for (auto &loc : locs) {
                    loc[0] -= groupOrigin[0];
                    loc[1] -= groupOrigin[1];
                }
            }

            // Create GT boxes in the right format (x, y, z, w, l, h, rot)
            std::vector<std::vector<float>> gtBoxes;
            for (size_t i = 0; i < locs.size(); ++i) {
                std::vector<float> box = {
                    locs[i][0], locs[i][1], locs[i][2],  // x, y, z
                    dims[i][1], dims[i][0], dims[i][2],  // w, l, h (swapped to match coordinate system)
                    rots[i]                             // rotation
                };
                gtBoxes.push_back(box);
            }

            // Filter by class names
            std::vector<size_t> keep;
            for (size_t i = 0; i < names.size(); ++i) {
                if (std::find(classNames.begin(), classNames.end(), names[i]) != classNames.end()) {
                    keep.push_back(i);
                }
            }

            // Apply class name filtering
            std::vector<int> filteredInstanceIds;
            std::vector<std::vector<float>> filteredGtBoxes;
            std::vector<std::string> filteredNames;
            std::vector<std::vector<float>> filteredVelocity;
            std::vector<bool> filteredValidFlag;
            std::vector<json> filteredBoxes2dFull;
            std::vector<json> filteredBoxes2dVisible;

            for (size_t idx : keep) {
                filteredInstanceIds.push_back(instanceIds[idx]);
                filteredGtBoxes.push_back(gtBoxes[idx]);
                filteredNames.push_back(names[idx]);
                filteredVelocity.push_back(velocity[idx]);
                filteredValidFlag.push_back(validFlag[idx]);
                filteredBoxes2dFull.push_back(boxes2dFull[idx]);
                filteredBoxes2dVisible.push_back(boxes2dVisible[idx]);
            }

            // Remove invisible annotations for sub scenes
            std::vector<size_t> visibilityKeep;
            std::vector<std::string> camsInGroup;
            for (const auto &[cam, _] : info["cams"].items()) {
                camsInGroup.push_back(cam);
            }

            for (size_t bid = 0; bid < filteredBoxes2dVisible.size(); ++bid) {
                const auto &visibleBoxes = filteredBoxes2dVisible[bid];
                bool isBoxVisibleInGroup = false;

                for (const auto &[cam, _] : visibleBoxes.items()) {
                    if (std::find(camsInGroup.begin(), camsInGroup.end(), cam) != camsInGroup.end()) {
                        isBoxVisibleInGroup = true;
                        break;
                    }
                }

                if (isBoxVisibleInGroup) {
                    visibilityKeep.push_back(bid);
                } else {
                    filteredValidFlag[bid] = false;
                }
            }

            // Store all processed information in the info object
            info["instance_inds"] = filteredInstanceIds;
            info["gt_boxes"] = filteredGtBoxes;
            info["gt_names"] = filteredNames;
            info["gt_velocity"] = filteredVelocity;
            info["valid_flag"] = filteredValidFlag;
            info["gt_boxes_2d"] = {
                {"full", filteredBoxes2dFull},
                {"visible", filteredBoxes2dVisible}
            };
        }

        infos.push_back(info);
    }

    return infos;
}

std::vector<Eigen::Matrix4f> transformLidarInfo(const std::vector<Eigen::Matrix4f> &lidar2img,
                                                AugConfig &augConfigs,
                                                int height, int width) {
    // Initialize variables with default values - following Python implementation exactly
    float resize = 1.0f;
    if (augConfigs.count("resize")) {
        resize = std::get<float>(augConfigs.at("resize"));
    }

    std::vector<int> resize_dims = {static_cast<int>(width * resize), static_cast<int>(height * resize)};

    // Default crop to full resized image [x_min, y_min, x_max, y_max]
    std::vector<int> crop = {0, 0, resize_dims[0], resize_dims[1]};
    if (augConfigs.count("crop")) {
         try {
            // Assuming C++ crop format matches Python [x_min, y_min, width, height]
            crop = std::get<std::vector<int>>(augConfigs.at("crop"));
            if (crop.size() != 4) {
                 std::cerr << "Warning: 'crop' vector size is not 4. Using default crop." << std::endl;
                 crop = {0, 0, resize_dims[0], resize_dims[1]}; // Reset to default
            }
         } catch (const std::bad_variant_access& e) {
             std::cerr << "Warning: Incorrect type for 'crop' in augConfigs. Using default crop." << std::endl;
             crop = {0, 0, resize_dims[0], resize_dims[1]}; // Reset to default
         } catch (const std::out_of_range& e) {
            std::cerr << "Warning: Error accessing 'crop' elements. Using default crop." << std::endl;
            crop = {0, 0, resize_dims[0], resize_dims[1]}; // Reset to default
         }
    }

    bool flip = false;
    if (augConfigs.count("flip")) {
        flip = std::get<bool>(augConfigs.at("flip"));
    }

    float rotate = 0.0f;
    if (augConfigs.count("rotate")) {
        rotate = std::get<float>(augConfigs.at("rotate")) / 180.0f * M_PI; // Convert degrees to radians
    }

    // Create the 3x3 transformation matrix (equivalent to Python's transform_matrix = torch.eye(3))
    Eigen::Matrix3f transformMatrix = Eigen::Matrix3f::Identity();

    // Scale by resize factor (in Python: transform_matrix[:2, :2] *= resize)
    transformMatrix.block<2, 2>(0, 0) *= resize;

    // Subtract crop offset from translation (Python: transform_matrix[:2, 2] -= crop[:2])
    // crop[0] is x_min, crop[1] is y_min
    transformMatrix.block<2, 1>(0, 2) -= Eigen::Vector2f(crop[0], crop[1]);

    // Apply horizontal flipping if enabled
    if (flip) {
        // flip_matrix = torch.tensor([[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]])
        // transform_matrix = flip_matrix @ transform_matrix
        Eigen::Matrix3f flipMatrix = Eigen::Matrix3f::Identity();
        flipMatrix(0, 0) = -1; // Flip horizontally
        // Ensure crop indices are valid before accessing
        if (crop.size() == 4) {
            flipMatrix(0, 2) = crop[2] - crop[0]; // Crop width offset after flipping
        } else {
             std::cerr << "Warning: Invalid crop dimensions for flip calculation." << std::endl;
        }
        transformMatrix = flipMatrix * transformMatrix;
    }

     // Apply rotation if needed (only if rotate is significantly different from 0)
    if (std::abs(rotate) > 1e-5)
    {
        // rot_matrix = torch.tensor([[cos(rotate), sin(rotate), 0], [-sin(rotate), cos(rotate), 0], [0, 0, 1]])
        Eigen::Matrix3f rotMatrix = Eigen::Matrix3f::Identity();
        rotMatrix(0, 0) = std::cos(rotate);
        rotMatrix(0, 1) = std::sin(rotate);
        rotMatrix(1, 0) = -std::sin(rotate);
        rotMatrix(1, 1) = std::cos(rotate);

        // Calculate rotation center (center of crop)
        // rot_center = torch.tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        // Ensure crop indices are valid before accessing
         Eigen::Vector2f rotCenter(0.0f, 0.0f);
         if (crop.size() == 4) {
             // crop[2]-crop[0] is crop width, crop[3]-crop[1] is crop height
            rotCenter << (crop[2] - crop[0]) / 2.0f, (crop[3] - crop[1]) / 2.0f;
         } else {
             std::cerr << "Warning: Invalid crop dimensions for rotation center calculation." << std::endl;
         }


        // Calculate translation for rotation
        // transformed_center = -rot_matrix[:2, :2] @ rot_center + rot_center
        // rot_matrix[:2, 2] = transformed_center
        Eigen::Vector2f transformedCenter = -rotMatrix.block<2, 2>(0, 0) * rotCenter + rotCenter;
        rotMatrix.block<2, 1>(0, 2) = transformedCenter;

        // Apply rotation to transformation matrix (transform_matrix = rot_matrix @ transform_matrix)
        transformMatrix = rotMatrix * transformMatrix;
    }


    // Create 4x4 transformation matrix (extend_matrix = torch.eye(4))
    Eigen::Matrix4f extendMatrix = Eigen::Matrix4f::Identity();
    extendMatrix.block<3, 3>(0, 0) = transformMatrix;

    // Apply the transformation to each lidar2img matrix
    std::vector<Eigen::Matrix4f> transformedLidar2Img;
    transformedLidar2Img.reserve(lidar2img.size()); // Pre-allocate memory

    for (const auto &mat : lidar2img) {
        // Perform standard 4x4 matrix multiplication, matching Python's `extend_matrix @ mat`
        Eigen::Matrix4f transformed = extendMatrix * mat;
        transformedLidar2Img.push_back(transformed);
    }
    return transformedLidar2Img;
}
