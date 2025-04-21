#include <stdio.h>
#include <math.h>
#include <ctime>
#include <cstring>
#include <cstdint>
#include <malloc.h>

#include "bagging.hpp"

void randomSampling(void* src_feature, void* src_label, void**& dst_feature, void**& dst_label,
  unsigned**& ori_feature_index, unsigned src_size, unsigned src_dim, unsigned dst_size,
  unsigned dst_dim, unsigned dst_count, unsigned feature_bytes, unsigned label_bytes) {
  srand(time(NULL));
  ori_feature_index = (unsigned**)malloc(dst_dim * sizeof(unsigned*));
  dst_feature = (void**)malloc(dst_count * sizeof(void*));
  dst_label = (void**)malloc(dst_count * sizeof(void*));
  unsigned* indexx = (unsigned*)malloc(dst_size * sizeof(int));
  for (unsigned i = 0; i < dst_count; ++i) {
    ori_feature_index[i] = (unsigned*)malloc(dst_dim * sizeof(unsigned));
    dst_feature[i] = malloc(dst_size * dst_dim * feature_bytes);
    dst_label[i] = malloc(dst_size * label_bytes);
    for (unsigned j = 0; j < dst_size; ++j) indexx[j] = rand() % src_size;
    for (unsigned j = 0; j < dst_dim; ++j) ori_feature_index[i][j] = rand() % src_dim;
    for (unsigned j = 0, J = 0; j < dst_size; ++j, J += dst_dim) {
      memcpy((char*)dst_label[i] + j * label_bytes,
        (char*)src_label + indexx[j] * label_bytes, label_bytes);
      indexx[j] *= src_dim;
      for (unsigned k = 0, K = J; k < dst_dim; ++k, ++K) {
        memcpy((char*)dst_feature[i] + K * feature_bytes,
          (char*)src_feature + (indexx[j] + ori_feature_index[i][k]) * feature_bytes,
          feature_bytes);
      }
    }
  }
  free(indexx);
}

void randomSampling4Bytes(void* src_feature, void* src_label, void**& dst_feature, void**& dst_label,
  unsigned**& ori_feature_index, unsigned src_size, unsigned src_dim, unsigned dst_size,
  unsigned dst_dim, unsigned dst_count) {
  srand(time(NULL));
  ori_feature_index = (unsigned**)malloc(dst_dim * sizeof(unsigned*));
  dst_feature = (void**)malloc(dst_count * sizeof(void*));
  dst_label = (void**)malloc(dst_count * sizeof(void*));
  unsigned* indexx = (unsigned*)malloc(dst_size * sizeof(int));
  for (unsigned i = 0; i < dst_count; ++i) {
    ori_feature_index[i] = (unsigned*)malloc(dst_dim * sizeof(unsigned));
    dst_feature[i] = malloc(dst_size * dst_dim * 4);
    dst_label[i] = malloc(dst_size * 4);
    for (unsigned j = 0; j < dst_size; ++j) indexx[j] = rand() % src_size;
    for (unsigned j = 0; j < dst_dim; ++j) ori_feature_index[i][j] = rand() % src_dim;
    for (unsigned j = 0, J = 0; j < dst_size; ++j, J += dst_dim) {
      ((uint32_t*)dst_label[i])[j] = ((uint32_t*)src_label)[indexx[j]];
      indexx[j] *= src_dim;
      for (unsigned k = 0, K = J; k < dst_dim; ++k, ++K)
        ((uint32_t*)dst_feature[i])[K] = ((uint32_t*)src_feature)[indexx[j] + ori_feature_index[i][k]];
    }
  }
  free(indexx);
}
/*
signed main(int argc, char* argv[]) {
  unsigned src_size = 100000;
  unsigned src_dim = 100;
  unsigned dst_size = 300;
  unsigned dst_dim = 32;
  unsigned dst_count = 10;
  unsigned feature_bytes = sizeof(float);
  unsigned label_bytes = sizeof(int);

  float* src_feature = (float*)malloc(src_size * src_dim * feature_bytes);
  int* src_label = (int*)malloc(src_size * label_bytes);

  for (unsigned i = 0; i < src_size; ++i) {
    for (unsigned j = 0; j < src_dim; ++j) {
      src_feature[i * src_dim + j] = static_cast<float>(rand()) / RAND_MAX;
    }
    src_label[i] = rand() % 2;
  }
  void** dst_feature;
  void** dst_label;
  unsigned** ori_feature_index;

  unsigned long long TimeBegin = clock(), TimeEnd;
  randomSampling(src_feature, src_label, dst_feature, dst_label,
    ori_feature_index, src_size, src_dim, dst_size, dst_dim,
    dst_count, feature_bytes, label_bytes);
  TimeEnd = clock();
  printf("Time %llu us\n", TimeEnd - TimeBegin);
  TimeBegin = clock();
  randomSampling4Bytes(src_feature, src_label, dst_feature, dst_label,
    ori_feature_index, src_size, src_dim, dst_size, dst_dim, dst_count);
  TimeEnd = clock();
  printf("Time %llu us\n", TimeEnd - TimeBegin);

  // Free allocated memory
  free(src_feature);
  free(src_label);

  for (unsigned i = 0; i < dst_count; ++i) {
    free(dst_feature[i]);
    free(dst_label[i]);
  }

  free(dst_feature);
  free(dst_label);

  return 0;
}
  */