#include <stdio.h>
#include <malloc.h>
#include <cstring>

struct featureMeta {
  unsigned Type;
  float Min, Max;
}
struct metaData {
  unsigned sample_count, class_count, feature_count;
  unsigned sample_byte, feature_byte;
  vector<featureMeta>
}

void read_config(metaData*& hee) {

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
    
    return data;
}

