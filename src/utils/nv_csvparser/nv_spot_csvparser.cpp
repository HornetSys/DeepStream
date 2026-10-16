/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unordered_map>
#include <deque>
#include <pthread.h>

#include <glib.h>
#include "nv_spot_csvparser.hpp"

#pragma GCC diagnostic ignored "-Wstrict-aliasing"

using namespace nvspot_csv;

#define DEFAULT_SPOT_CSV_FIELDS 80

SpotCSVParser::~SpotCSVParser()
{
  this->csvFileName.clear();
  this->CameraViews.clear();
  this->CSVMap.clear();
}

SpotCSVParser::SpotCSVParser(string CSVFileName)
{
  csvFileName.assign(CSVFileName);
  LoadCSVData();
}

static void
get_spot_csv_tokens (const string &text, vector<string> &tokens)
{
  /* This is based on assumption that fields and their locations
   * are fixed in CSV file. This should be updated accordingly if
   * that is not the case.
   */
  gint count = 0;

  gchar **csv_tokens = g_strsplit (text.c_str(), ",", -1);
  gchar **temp = csv_tokens;
  gchar *token;

  while (*temp && count < DEFAULT_SPOT_CSV_FIELDS) {
    token = *temp++;
    tokens.push_back (string(g_strstrip(token)));
    count++;
  }
  g_strfreev (csv_tokens);
}

void SpotCSVParser::LoadCSVData()
{
  int firstline = 1;
  int i = 0;
  int rowId = 0;
  NvSpotCsvFields fields = {0};
  _SpotIndexMap_Result  SpotIndexMap_result;
  _SpotCSVMap_Result CSVMap_result;
  string temp_str;
  std::vector<std::string> cameraId_vec;

  //cout << "SPOT Calibration File : " << csvFileName.c_str() << endl;

  ifstream inFile (csvFileName.c_str());

  if (!inFile.good())
  {
    cout << "Invalid Spot Calibration File Provided. " <<
        csvFileName.c_str() << " Exiting...\n" << endl;
    exit(-1);
  }

  std::string line = "";

  try
  {
    // Iterate through each line and split the content using delimeter
    while (getline(inFile, line))
    {
      std::vector<std::string> vec;

      if (firstline == 1)
      {
        firstline = 0;
        continue;
      }

      get_spot_csv_tokens (line, vec);
      {
        i = 0;

        fields.rowId = rowId++;
        fields.cameraId = atoi (vec.at(i++).c_str());
        fields.sensorId.assign(vec.at(i++));

        // Ignore camDesc field.
        i++;

        fields.cameraIdString.assign(vec.at(i++));

        // Extract CameraID from CameraID-String
        //fields.cameraId = atoi (cameraId_vec.at(3).c_str());

        fields.spotId.assign(vec.at(i++));

        // Ignore type field
        i++;

        fields.level.assign(vec.at(i++));
        fields.surfaceid = atoi (vec.at(i++).c_str());
        fields.spot_index = atoi (vec.at(i++).c_str());

        fields.dewarpTopAngle = atof (vec.at(i++).c_str());
        fields.dewarpBottomAngle = atof (vec.at(i++).c_str());
        fields.dewarpPitch = atof (vec.at(i++).c_str());
        fields.dewarpYaw = atof (vec.at(i++).c_str());
        fields.dewarpRoll = atof (vec.at(i++).c_str());

        fields.vertical_left = atoi (vec.at(i++).c_str());
        fields.vertical_right = atoi (vec.at(i++).c_str());

        fields.Horizon_x1 = atoi (vec.at(i++).c_str());
        fields.Horizon_y1 = atoi (vec.at(i++).c_str());
        fields.Horizon_x2 = atoi (vec.at(i++).c_str());
        fields.Horizon_y2 = atoi (vec.at(i++).c_str());
        fields.spot_roi_x1 = atoi (vec.at(i++).c_str());
        fields.spot_roi_y1 = atoi (vec.at(i++).c_str());
        fields.spot_roi_x2 = atoi (vec.at(i++).c_str());
        fields.spot_roi_y2 = atoi (vec.at(i++).c_str());

        fields.x0 = atof (vec.at(i++).c_str());
        fields.y0 = atof (vec.at(i++).c_str());
        fields.x1 = atof (vec.at(i++).c_str());
        fields.y1 = atof (vec.at(i++).c_str());
        fields.x2 = atof (vec.at(i++).c_str());
        fields.y2 = atof (vec.at(i++).c_str());
        fields.x3 = atof (vec.at(i++).c_str());
        fields.y3 = atof (vec.at(i++).c_str());
#if 0
        fields.lng0 = atof (vec.at(i++).c_str());
        fields.lat0 = atof (vec.at(i++).c_str());
        fields.lng1 = atof (vec.at(i++).c_str());
        fields.lat1 = atof (vec.at(i++).c_str());
        fields.lng2 = atof (vec.at(i++).c_str());
        fields.lat2 = atof (vec.at(i++).c_str());
        fields.lng3 = atof (vec.at(i++).c_str());
        fields.lat3 = atof (vec.at(i++).c_str());
#endif
        fields.dewarpFocalLength = atof (vec.at(i++).c_str());
        fields.dewarpWidth = atoi (vec.at(i++).c_str());
        fields.dewarpHeight = atoi (vec.at(i++).c_str());

        auto map_entry = CSVMap.find (fields.cameraId);
        if (map_entry == CSVMap.end())
        {
          _SpotIndex_Map temp;
          SpotIndexMap_result = temp.insert (make_pair (make_pair(fields.surfaceid, fields.spot_index), fields));
          if (SpotIndexMap_result.second == false)
          {
            cout << " SpotCSV: " << __LINE__ << " : Entry insert failed for camera-id: " <<  fields.cameraId
                << " For SPOT Index: " << fields.spot_index << endl;
            exit (-1);
          }
          CSVMap_result = CSVMap.insert (make_pair(fields.cameraId, temp));
          //cout << " Adding First Entry for camera  " <<  fields.cameraId << " " << fields.surfaceid << " " << fields.spot_index << endl;
          //cout << " SpotMapIndex "  << &temp << " Size " << temp.size() << endl;
        }
        else
        {
          _SpotIndex_Map temp = (_SpotIndex_Map) map_entry->second;
          //cout << " Temp "  << &temp << " Size " << temp.size() << endl;
          SpotIndexMap_result = temp.insert (make_pair (make_pair(fields.surfaceid, fields.spot_index), fields));
          if (SpotIndexMap_result.second == false)
          {
            cout << " SpotCSV: " << __LINE__ << " : Entry insert failed for camera-id: " <<  fields.cameraId
                << " For SPOT Index: " << fields.spot_index << endl;
            exit (-1);
          }
          //cout << "        Temp "  << &temp << " Size " << temp.size() << endl;

          // TODO : We should not erase, instead insert should work
          CSVMap.erase (fields.cameraId);
          CSVMap_result = CSVMap.insert (make_pair(fields.cameraId, temp));
        }
      }
    }
  }
  catch(const std::runtime_error& rerr)
  {
    std::cerr << "SpotCSV: Runtime error: " << rerr.what() << std::endl;
  }
  catch(const std::exception& exp)
  {
    std::cerr << "SpotCSV: Error occurred: " << exp.what() << std::endl;
  }
  catch(...)
  {
    std::cerr << "SpotCSV: Unknown failure occurred." << std::endl;
  }

  if (0)
  {
    std::vector<NvSpotCsvFields> csvSpotData;
    getNvSpotCSVData (csvSpotData);
    NvSpotCsvFields values = csvSpotData.at(0);
    printSpotCSVFields (values);
  }


  // Close the File
  inFile.close();

}

void SpotCSVParser::getNvSpotCSVData(std::vector<NvSpotCsvFields> &csvSpotData)
{
  for (auto map_entry = CSVMap.begin(); map_entry != CSVMap.end(); ++map_entry)
  {
    _SpotIndex_Map &temp = (_SpotIndex_Map&) map_entry->second;
    for (auto spot_entry = temp.begin(); spot_entry != temp.end(); ++spot_entry)
    {
      NvSpotCsvFields fields = (NvSpotCsvFields) spot_entry->second;
      csvSpotData.push_back (fields);
    }
  }
}

uint32_t SpotCSVParser::getNvSpotCSVMaxViews (uint32_t cam_id, vector<int>* vector_surface_index)
{
  uint32_t max_views = 0;

  auto map_entry = CameraViews.find (cam_id);
  if (map_entry == CameraViews.end())
  {
    max_views = prepareSpotCSVMaxViews(cam_id, vector_surface_index);

    if (max_views)
    {
      // Create pair of camera-id and num_spot_view
      _SpotCameraView_Result cameraView_result = CameraViews.insert (make_pair (cam_id, max_views));
      if (cameraView_result.second == false)
      {
        cout << " SpotCSV: CameraViews " << __LINE__ << " : Entry insert failed for camera-id: " <<  cam_id << " Views: " << max_views << endl;
        exit (-1);
      }
    }
  }
  else
  {
    max_views = map_entry->second;
  }
  //cout << "SpotCSV: " << __func__ << " Max SPOT Views for CameraId: " << cam_id << " is " << max_views << endl;
  return max_views;
}

uint32_t SpotCSVParser::prepareSpotCSVMaxViews (uint32_t cam_id, vector<int>* vector_surface_index)
{
  _SpotIndex_Map::iterator iter;
  int32_t surface_id = -1;
  int32_t num_views = 0;

  auto map_entry = CSVMap.find (cam_id);
  if (map_entry == CSVMap.end())
  {
    cout << "SpotCSV: Camera entry NOT found. CameraId: " << cam_id << endl;
    return num_views;
  }
  _SpotIndex_Map &temp = (_SpotIndex_Map &) map_entry->second;

  for (iter = temp.begin(); iter != temp.end(); iter++)
  {
    std::pair<int, int> &surfaceid_type_pair = (std::pair<int, int>  &) iter->first;

    if (surface_id != surfaceid_type_pair.first)
    {
      //int32_t spot_index = 0;
      surface_id = surfaceid_type_pair.first;
      //spot_index = surfaceid_type_pair.second;
      vector_surface_index->push_back(surface_id);
      num_views++;
    }
    //NvSpotCsvFields fields = iter->second;
  }
  //cout << " Num View Are " << num_views << " in function " << __func__ << endl;
  return num_views;
}

int SpotCSVParser::getNvSpotCSVFields (uint32_t cam_id, uint32_t surface_id, uint32_t spot_id, NvSpotCsvFields *fields)
{
  auto map_entry = CSVMap.find (cam_id);

  if (map_entry == CSVMap.end())
  {
    cout << "SpotCSV: Camera entry NOT found. CameraId: " << cam_id << endl;
    return -1;
  }

  _SpotIndex_Map &temp = (_SpotIndex_Map&) map_entry->second;
  std::pair<uint32_t, uint32_t> new_pair = make_pair(surface_id, spot_id);

  auto spot_entry = temp.find (new_pair);
  if (spot_entry == temp.end())
  {
    cout << "SpotCSV: Camera entry found. CameraId: " << cam_id << " found " << endl;
    return -1;
  }

  NvSpotCsvFields values = (NvSpotCsvFields) spot_entry->second;

  *fields = values;
  return 0;

  //cout << map_entry->first << " " << spot_entry->first.first << " " << spot_entry->first.second << " " << endl;
  //printSpotCSVFields(fields);
}

void SpotCSVParser::printNvSpotCSVData(void)
{
  for (auto map_entry = CSVMap.begin(); map_entry != CSVMap.end(); ++map_entry)
  {
    _SpotIndex_Map &temp = (_SpotIndex_Map&) map_entry->second;

    for (auto spot_entry = temp.begin(); spot_entry != temp.end(); ++spot_entry)
    {
      cout << map_entry->first << " " << spot_entry->first.first <<  " " << spot_entry->first.second << endl;
    }
  }
}


void SpotCSVParser::printSpotCSVFields(NvSpotCsvFields val)
{
  cout << "------------ " << __func__  << " -----------" << endl;
  cout << val.cameraId << endl;
  cout << val.spotId << endl;
  cout << val.vertical_left << " " << val.vertical_right << endl;
  cout << "x0,y0 " << val.x0 << " " << val.y0 << " x1,y1 " << val.x1 << " " << val.y1 << endl;
  cout << val.lng0 << endl;
  cout << val.lng1 << endl;
  cout << " ----------------"  << endl;
}

void SpotCSVParser::printNvSpotCSVData(uint32_t cam_id, uint32_t surface_id, uint32_t spot_id)
{
  auto map_entry = CSVMap.find (cam_id);
  cout << "-------------------------" << endl;
  if (map_entry == CSVMap.end())
  {
    cout << "SpotCSV: Camera entry NOT found. CameraId: " << cam_id << endl;
    return;
  }

  _SpotIndex_Map &temp = (_SpotIndex_Map &) map_entry->second;
  auto spot_entry = temp.find (make_pair(surface_id, spot_id));
  if (spot_entry == temp.end())
  {
    cout << "SpotCSV: Camera entry found. CameraId: " << cam_id << " found " << endl;
    return;
  }

  NvSpotCsvFields fields = (NvSpotCsvFields) spot_entry->second;

  cout << map_entry->first << " " << spot_entry->first.first << " " << spot_entry->first.second << " " << endl;
  printSpotCSVFields(fields);
}

