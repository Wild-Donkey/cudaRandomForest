#include <stdio.h>
#include <math.h>
#include <ctime>
#include <cstring>


void randomSampling(void* src_feature, void * src_label, void **& dst_feature, void **& dst_label, 
  unsigned src_size, unsigned src_dim, unsigned dst_size, unsigned dst_dim, unsigned dst_count, 
  unsigned feature_bytes, unsigned label_bytes);
void randomSampling4Bytes(void* src_feature, void * src_label, void ** &dst_feature, void ** &dst_label, 
  unsigned src_size, unsigned src_dim, unsigned dst_size, unsigned dst_dim, unsigned dst_count);