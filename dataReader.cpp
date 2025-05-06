#include <stdio.h>
#include <cstring>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "dataReader.hpp"

metaData::metaData() {}
metaData::~metaData() {
  features_meta.clear();
}
featureMeta::featureMeta() {}
featureMeta::~featureMeta() {
  stringSet.clear();
}

void read_config(metaData*& head, const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("Cant open File " + filename);
  head = new metaData();
  file >> head->sample_count >> head->feature_count;


  for (unsigned i = 0; i < head->feature_count; ++i) {
    featureMeta TmpF;
    file >> TmpF.Type;
    if (TmpF.Type == 2) {
      unsigned Sz;
      file >> Sz;
      std::string TmpS;
      for (unsigned j = 0; j < Sz; ++j) file >> TmpS, TmpF.stringSet[TmpS] = j;
    } else {
      file >> TmpF.Min >> TmpF.Max;
    }
    head->features_meta.push_back(TmpF);
  }
  file.close();
}

std::vector<std::vector<std::string>> read_csv(const std::string& filename) {
  std::vector<std::vector<std::string>> data;
  std::ifstream file(filename);

  if (!file.is_open())
    throw std::runtime_error("Cant open File " + filename);

  std::string line;
  while (std::getline(file, line)) {
    std::vector<std::string> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
      row.push_back(cell);
    }

    data.push_back(row);
  }
  file.close();
  return data;
}

void get_data(float* data, bool* is_category_feature, const metaData& head, std::vector<std::vector<std::string>> file_data) {
  unsigned sample_size = head.sample_count;
  unsigned feature_count = head.feature_count;
  for (unsigned i = 0; i < feature_count; ++i) {
    is_category_feature[i] = (head.features_meta[i].Type == 2);
    for (unsigned j = 0, J = i; j < sample_size; ++j, J += feature_count) {
      if (head.features_meta[i].Type == 2) {
        data[J] = head.features_meta[i].stringSet.at(file_data[j + 1][i]);
      } else {
        data[J] = std::stof(file_data[j + 1][i]);
      }
    }
  }
}
