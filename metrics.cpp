#include <stdio.h>
#include <math.h>
#include <ctime>
#include <cstring>

const unsigned dataSize = 1000000;
const unsigned classCount = 10;

float computeEntropy(unsigned* x, unsigned sample_count, unsigned class_count) {
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
float informationGain(float* x, unsigned* y, unsigned sample_count, unsigned class_count) {
  float entropy = computeEntropy(y, n, class_count);

  for (unsigned i = 0; i < n; ++i) {
    if(x) 
  }
  return entropy - entropy_y;
}

signed main() {
  unsigned *h_label = (unsigned *)malloc(dataSize * sizeof(unsigned));
  
  printf("Done");
  // 生成随机数据 (0-1之间)
  for (unsigned i = 0; i < dataSize; i++) 
    for (unsigned j = 0; j < n; ++j) h_label[i] = (float)rand() / RAND_MAX;
  for (unsigned i = 0; i < dataSize; i++) h_label[i] = rand() % numBins;
  
  // 计算熵
  unsigned long long TimeBegin = clock(), TimeEnd;
  float entropy = computeEntropy(h_label, dataSize, numBins);
  TimeEnd = clock();
  printf("Entropy: %f Time %llu us\n", entropy, TimeEnd - TimeBegin);
  

  return 0;
}