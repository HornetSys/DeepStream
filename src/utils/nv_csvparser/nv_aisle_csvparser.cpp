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

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <deque>
#include <pthread.h>

#include <glib.h>
#include "nv_aisle_csvparser.hpp"

#pragma GCC diagnostic ignored "-Wstrict-aliasing"

using namespace nvaisle_csv;

#define DEFAULT_AISLE_CSV_FIELDS 80

AisleCSVParser::~AisleCSVParser()
{
  this->csvFileName.clear();
  this->CameraViews.clear();
  this->CSVMap.clear();
}

AisleCSVParser::AisleCSVParser(string CSVFileName)
{
  csvFileName.assign(CSVFileName);
  LoadCSVData();
}

static void
get_aisle_csv_tokens (const string &text, vector<string> &tokens)
{
  /* This is based on assumption that fields and their locations
   * are fixed in CSV file. This should be updated accordingly if
   * that is not the case.
   */
  gint count = 0;

  gchar **csv_tokens = g_strsplit (text.c_str(), ",", -1);
  gchar **temp = csv_tokens;
  gchar *token;

  while (*temp && count < DEFAULT_AISLE_CSV_FIELDS) {
    token = *temp++;
    tokens.push_back (string(g_strstrip(token)));
    count++;
  }
  g_strfreev (csv_tokens);
}

void AisleCSVParser::LoadCSVData()
{
  int firstline = 1;
  int i = 0;
  int pos = -1;
  int rowId = 0;
  NvAisleCsvFields fields;
  _AisleIndexMap_Result  AisleIndexMap_result;
  _AisleCSVMap_Result CSVMap_result;
  string temp_str;
  std::vector<std::string> cameraId_vec;
  std::string aisle_extracted_str;

  //cout << "AisleCSV: Aisle Calibration File : " << csvFileName.c_str() << endl;

  ifstream inFile (csvFileName.c_str());

  if (!inFile.good())
  {
    cout << "AisleCSV: Invalid Aisle Calibration File " << csvFileName.c_str()
        << " Provided. Exiting...\n" << endl;
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

      get_aisle_csv_tokens (line, vec);

      {
        i = 0;
        fields.rowId = rowId++;
        fields.cameraId = atoi (vec.at(i++).c_str());
        fields.sensorId.assign(vec.at(i++));

        // Ignore camDesc field.
        i++;

        // Extract CameraID from CameraID-String
        //fields.cameraId = atoi (cameraId_vec.at(3).c_str());

        fields.cameraIDString.assign(vec.at(i++));
        fields.aisleIdStr.assign(vec.at(i++));
        // Ignore aisleName field.
        i++;
        fields.level.assign(vec.at(i++));

        fields.aisleId = -1;

        pos = fields.aisleIdStr.find("A0");
        if (pos >= 0)
        {
          aisle_extracted_str = fields.aisleIdStr.substr (pos);
          if (aisle_extracted_str.compare("A0") == 0)
          {
            fields.aisleId = 0;
          }
        }
        else
        {
          pos = fields.aisleIdStr.find("A1");
          if (pos >=0 )
          {
            aisle_extracted_str = fields.aisleIdStr.substr (pos);
            if (aisle_extracted_str.compare("A1") == 0)
            {
              fields.aisleId = 1;
            }
          }
        }

        fields.dewarpTopAngle = atof (vec.at(i++).c_str());
        fields.dewarpBottomAngle = atof (vec.at(i++).c_str());
        fields.dewarpPitch = atof (vec.at(i++).c_str());
        fields.dewarpYaw = atof (vec.at(i++).c_str());
        fields.dewarpRoll = atof (vec.at(i++).c_str());

        fields.numROIpoints = atoi (vec.at(i++).c_str());

        fields.ROI_x0 = atof (vec.at(i++).c_str());
        fields.ROI_y0 = atof (vec.at(i++).c_str());
        fields.ROI_x1 = atof (vec.at(i++).c_str());
        fields.ROI_y1 = atof (vec.at(i++).c_str());
        fields.ROI_x2 = atof (vec.at(i++).c_str());
        fields.ROI_y2 = atof (vec.at(i++).c_str());
        fields.ROI_x3 = atof (vec.at(i++).c_str());
        fields.ROI_y3 = atof (vec.at(i++).c_str());
        fields.ROI_x4 = atof (vec.at(i++).c_str());
        fields.ROI_y4 = atof (vec.at(i++).c_str());
        fields.ROI_x5 = atof (vec.at(i++).c_str());
        fields.ROI_y5 = atof (vec.at(i++).c_str());
        fields.ROI_x6 = atof (vec.at(i++).c_str());
        fields.ROI_y6 = atof (vec.at(i++).c_str());
        fields.ROI_x7 = atof (vec.at(i++).c_str());
        fields.ROI_y7 = atof (vec.at(i++).c_str());

        fields.gx0 = atof (vec.at(i++).c_str());
        fields.gy0 = atof (vec.at(i++).c_str());
        fields.gx1 = atof (vec.at(i++).c_str());
        fields.gy1 = atof (vec.at(i++).c_str());
        fields.gx2 = atof (vec.at(i++).c_str());
        fields.gy2 = atof (vec.at(i++).c_str());
        fields.gx3 = atof (vec.at(i++).c_str());
        fields.gy3 = atof (vec.at(i++).c_str());

        fields.cx0 = atof (vec.at(i++).c_str());
        fields.cy0 = atof (vec.at(i++).c_str());
        fields.cx1 = atof (vec.at(i++).c_str());
        fields.cy1 = atof (vec.at(i++).c_str());
        fields.cx2 = atof (vec.at(i++).c_str());
        fields.cy2 = atof (vec.at(i++).c_str());
        fields.cx3 = atof (vec.at(i++).c_str());
        fields.cy3 = atof (vec.at(i++).c_str());

        fields.h0 = atof (vec.at(i++).c_str());;
        fields.h1 = atof (vec.at(i++).c_str());;
        fields.h2 = atof (vec.at(i++).c_str());;
        fields.h3 = atof (vec.at(i++).c_str());;
        fields.h4 = atof (vec.at(i++).c_str());;
        fields.h5 = atof (vec.at(i++).c_str());;
        fields.h6 = atof (vec.at(i++).c_str());;
        fields.h7 = atof (vec.at(i++).c_str());;
        fields.h8 = atof (vec.at(i++).c_str());;

        fields.dewarpFocalLength = atof (vec.at(i++).c_str());
        fields.dewarpWidth = atoi (vec.at(i++).c_str());
        fields.dewarpHeight = atoi (vec.at(i++).c_str());

        fields.entry = atoi (vec.at(i++).c_str());
        fields.entry_ROI_x0 = atof (vec.at(i++).c_str());
        fields.entry_ROI_y0 = atof (vec.at(i++).c_str());
        fields.entry_ROI_x1 = atof (vec.at(i++).c_str());
        fields.entry_ROI_y1 = atof (vec.at(i++).c_str());
        fields.entry_ROI_x2 = atof (vec.at(i++).c_str());
        fields.entry_ROI_y2 = atof (vec.at(i++).c_str());
        fields.entry_ROI_x3 = atof (vec.at(i++).c_str());
        fields.entry_ROI_y3 = atof (vec.at(i++).c_str());

        fields.exit = atoi (vec.at(i++).c_str());
        fields.exit_ROI_x0 = atof (vec.at(i++).c_str());
        fields.exit_ROI_y0 = atof (vec.at(i++).c_str());
        fields.exit_ROI_x1 = atof (vec.at(i++).c_str());
        fields.exit_ROI_y1 = atof (vec.at(i++).c_str());
        fields.exit_ROI_x2 = atof (vec.at(i++).c_str());
        fields.exit_ROI_y2 = atof (vec.at(i++).c_str());
        fields.exit_ROI_x3 = atof (vec.at(i++).c_str());
        fields.exit_ROI_y3 = atof (vec.at(i++).c_str());

        auto map_entry = CSVMap.find (fields.cameraId);
        if (map_entry == CSVMap.end())
        {
          _AisleIndex_Map temp;
          AisleIndexMap_result = temp.insert (make_pair (fields.aisleId, fields));
          if (AisleIndexMap_result.second == false)
          {
            cout << "AisleCSV: CameraId: " << fields.cameraId << " Entry insert failed " << " For AisleID: " << fields.aisleId << endl;
            exit (-1);
          }
          CSVMap_result = CSVMap.insert (make_pair(fields.cameraId, temp));
          //cout << " Adding First Entry for camera  " <<  fields.cameraId << " " << fields.aisleId << " " << endl;
          //cout << " AisleMapIndex "  << &temp << " Size " << temp.size() << endl;
        }
        else
        {
          _AisleIndex_Map temp = (_AisleIndex_Map) map_entry->second;
          //cout << " Temp "  << &temp << " Size " << temp.size() << endl;
          AisleIndexMap_result = temp.insert (make_pair (fields.aisleId, fields));
          if (AisleIndexMap_result.second == false)
          {
            cout << "AisleCSV: CameraId: " << fields.cameraId << " Duplicate: Entry Insert failed " << " For AisleID: " << fields.aisleId << endl;
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
    std::cerr << "AisleCSV: Runtime error: " << rerr.what() << std::endl;
  }
  catch(const std::exception& exp)
  {
    std::cerr << "AisleCSV: Error occurred: " << exp.what() << std::endl;
  }
  catch(...)
  {
    std::cerr << "AisleCSV: Unknown failure occurred." << std::endl;
  }

  if (0)
  {
    std::vector<NvAisleCsvFields> csvAisleData;
    getNvAisleCSVData (csvAisleData);
    NvAisleCsvFields values = csvAisleData.at(0);
    printAisleCSVFields (values);
  }

  // Close the File
  inFile.close();

}

void AisleCSVParser::getNvAisleCSVData(std::vector<NvAisleCsvFields> &csvAisleData)
{
  for (auto map_entry = CSVMap.begin(); map_entry != CSVMap.end(); ++map_entry)
  {
    _AisleIndex_Map &temp = (_AisleIndex_Map&) map_entry->second;
    for (auto aisle_entry = temp.begin(); aisle_entry != temp.end(); ++aisle_entry)
    {
      NvAisleCsvFields fields = (NvAisleCsvFields) aisle_entry->second;
      csvAisleData.push_back (fields);
    }
  }
}

uint32_t AisleCSVParser::getNvAisleCSVMaxViews (uint32_t cam_id, vector<int>* vector_surface_index)
{
  uint32_t max_views = 0;

  auto map_entry = CameraViews.find (cam_id);
  if (map_entry == CameraViews.end())
  {
    max_views = prepareAisleCSVMaxViews(cam_id, vector_surface_index);

    if (max_views)
    {
      // Create pair of camera-id and num_aisle_view
      _AisleCameraView_Result cameraView_result = CameraViews.insert (make_pair (cam_id, max_views));
      if (cameraView_result.second == false)
      {
        cout << " AisleCSV: CameraViews " << __LINE__ << " : Entry insert failed for camera-id: " <<  cam_id << " Views: " << max_views << endl;
        exit (-1);
      }
    }
  }
  else
  {
    max_views = map_entry->second;
  }
  //cout << "AisleCSV: " << __func__ << " Max AISLE Views for CameraId: " << cam_id << " is " << max_views << endl;
  return max_views;
}

uint32_t AisleCSVParser::prepareAisleCSVMaxViews (uint32_t cam_id, vector<int>* vector_surface_index)
{
  _AisleIndex_Map::iterator iter;
  int32_t last_aisle = -1;
  int32_t aisle_type;
  int32_t num_views = 0;

  auto map_entry = CSVMap.find (cam_id);
  if (map_entry == CSVMap.end())
  {
    cout << "AisleCSV: Camera entry NOT found. CameraId: " << cam_id << endl;
    return num_views;
  }

  _AisleIndex_Map &temp = (_AisleIndex_Map &) map_entry->second;
  for (iter = temp.begin(); iter != temp.end(); iter++)
  {
    aisle_type = iter->first;
    if (last_aisle != aisle_type)
    {
      last_aisle = aisle_type;
      vector_surface_index->push_back(aisle_type);
      num_views++;
    }
  }
  //cout << " Num View Are " << num_views << " in function " << __func__ << endl;
  return num_views;
}

int AisleCSVParser::getNvAisleCSVFields (uint32_t cam_id, uint32_t aisle_id, NvAisleCsvFields *fields)
{
  auto map_entry = CSVMap.find (cam_id);

  if (map_entry == CSVMap.end())
  {
    cout << "AisleCSV: CameraID " << cam_id << " Entry NOT found. Exiting " << endl;
    return -1;
  }
  _AisleIndex_Map &temp = (_AisleIndex_Map&) map_entry->second;
  auto aisle_entry = temp.find (aisle_id);
  if (aisle_entry == temp.end())
  {
    cout << "AisleCSV: Aisle entry NOT found for CameraID " << cam_id << " found " << endl;
    return -1;
  }

  NvAisleCsvFields values = (NvAisleCsvFields) aisle_entry->second;

  *fields = values;
  return 0;

  //cout << map_entry->first << " " << aisle_entry->first.first << " " << aisle_entry->first.second << " " << endl;
  //printAisleCSVFields(fields);
}

void AisleCSVParser::printNvAisleCSVData(void)
{
  for (auto map_entry = CSVMap.begin(); map_entry != CSVMap.end(); ++map_entry)
  {
    _AisleIndex_Map &temp = (_AisleIndex_Map&) map_entry->second;

    for (auto aisle_entry = temp.begin(); aisle_entry != temp.end(); ++aisle_entry)
    {
      cout << map_entry->first << " " << aisle_entry->first <<  endl;
    }
  }
}


void AisleCSVParser::printAisleCSVFields(NvAisleCsvFields val)
{
  cout << "------------ " << __func__  << " -----------" << endl;
  cout << val.cameraId << endl;
  cout << val.aisleId << endl;
  cout << "gx0,gy0 " << val.gx0 << " " << val.gy0 << " gx1,gy1 " << val.gx1 << " " << val.gy1 << endl;
  cout << "cx0,cy0 " << val.cx0 << " " << val.cy0 << " cx1,cy1 " << val.cx1 << " " << val.cy1 << endl;
  cout << " ----------------"  << endl;
}

void AisleCSVParser::printNvAisleCSVData(uint32_t cam_id, uint32_t aisle_id)
{
  auto map_entry = CSVMap.find (cam_id);
  cout << "-------------------------" << endl;
  if (map_entry == CSVMap.end())
  {
    cout << "AisleCSV: CameraID " << cam_id << " Entry NOT found. Exiting " << endl;
    return;
  }

  _AisleIndex_Map &temp = (_AisleIndex_Map&) map_entry->second;
  auto aisle_entry = temp.find (aisle_id);
  if (aisle_entry == temp.end())
  {
    cout << "AisleCSV: Aisle entry NOT found for CameraID " << cam_id << " found " << endl;
    return;
  }

  NvAisleCsvFields fields = (NvAisleCsvFields) aisle_entry->second;

  cout << map_entry->first << " " << aisle_entry->first << " " << endl;
  printAisleCSVFields(fields);
}

