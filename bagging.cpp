#include <stdio.h>
#include <math.h>
#include <ctime>
#include <cstring>

const unsigned dataSize = 4000000;
const unsigned numBins = 2; // 直方图bin数量

float computeEntropy(unsigned* x, unsigned n, unsigned class_count) {
  float entropy = 0.0f;
  float hist[class_count];
  
  memset(hist, 0, sizeof(hist));
  for (unsigned i = 0; i < n; i++) hist[x[i]] += 1;
  for (unsigned i = 0; i < class_count; i++) {
    hist[i] /= n;
    entropy -= hist[i] * log2(hist[i]);
  }
  return entropy;
}

signed main() {
  unsigned h_data[dataSize];
  
  printf("Done");
  // 生成随机数据 (0-1之间)
  for (unsigned i = 0; i < dataSize; i++) h_data[i] = rand() % numBins;
  
  // 计算熵
  unsigned long long TimeBegin = clock(), TimeEnd;
  float entropy = computeEntropy(h_data, dataSize, numBins);
  TimeEnd = clock();
  printf("信息熵: %f Time %llu\n", entropy, TimeEnd - TimeBegin);
  

  return 0;
}