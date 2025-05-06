#include <stdio.h>
#include <malloc.h>
#include <cstring>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

struct featureMeta {
  featureMeta();
  ~featureMeta();
  unsigned Type; //0 for float, 1 for int, 2 for string
  float Min = 0, Max = 0; //if float or int
  std::map<std::string, unsigned> stringSet;
};
struct metaData {
  metaData();
  ~metaData();
  unsigned sample_count, feature_count;
  unsigned feature_byte = 4, label_byte = 4;
  std::vector<featureMeta> features_meta;
};

void read_config(metaData*& head, const std::string& filename);
std::vector<std::vector<std::string>> read_csv(const std::string& filename);
void get_data(float* data, bool* is_category_feature, const metaData& head, std::vector<std::vector<std::string>> file_data);
