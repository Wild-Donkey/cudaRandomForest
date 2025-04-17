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