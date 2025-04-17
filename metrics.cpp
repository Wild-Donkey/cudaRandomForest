#include <stdio.h>
#include <math.h>
#include <ctime>
#include <cstring>
#include <algorithm>

const unsigned dataSize = 1000000;
const unsigned dataDim = 1;
const unsigned classCount = 10;

void computeHistogram(unsigned* label, unsigned*& hist, unsigned sample_count, unsigned class_count) {
  hist = (unsigned*)malloc(class_count * sizeof(unsigned));
  memset(hist, 0, class_count * sizeof(unsigned));
  
  for (unsigned i = 0; i < sample_count; i++) ++hist[label[i]];
}
float computeEntropyWithHist(unsigned* hist, unsigned sample_count, unsigned class_count) {
  if(sample_count == 0) return 0.0f;
  float entropy = 0.0f;
  
  for (unsigned i = 0; i < class_count; i++) {
    float probab = (float)hist[i] / sample_count;
    entropy -= probab * log2(probab);
  }
  return entropy;
}
float computeEntropy(unsigned* label, unsigned sample_count, unsigned class_count) {
  if(sample_count == 0) return 0.0f;
  unsigned* hist;
  computeHistogram(label, hist, sample_count, class_count);
  float entropy = computeEntropyWithHist(hist, sample_count, class_count);
  free(hist);
  return entropy;
}

std::pair<float, float> informationGain(float* feature, unsigned* label, unsigned sample_count, unsigned class_count) {
  if(sample_count == 0) return {0.0f, 0.0f};

  float entropy, entropy_y, splitPoint;
  float entropy_l, entropy_r;
  std::pair<float, unsigned> *f_l = (std::pair<float, unsigned> *)malloc(sample_count * sizeof(std::pair<float, unsigned>));
  unsigned *histl = (unsigned *)malloc(class_count * sizeof(unsigned)), *histr;
  memset(histl, 0, class_count * sizeof(unsigned));
  computeHistogram(label, histr, sample_count, class_count);

  for (unsigned i = 0; i < sample_count; ++i) f_l[i] = {feature[i], label[i]};
  std::sort(f_l, f_l + sample_count);

  entropy_r = entropy = computeEntropyWithHist(histr, sample_count, class_count);
  entropy_l = 0;
  entropy_y = entropy_l + entropy_r;
  splitPoint = f_l[0].first - 0.001f;
  
  for (unsigned i = 0; i < sample_count - 1; ++i) {
    histl[f_l[i].second]++;
    histr[f_l[i].second]--;
    if(f_l[i].first == f_l[i + 1].first) continue;
    entropy_l = ((i + 1.0f) / sample_count) * computeEntropyWithHist(histl, i + 1, class_count);
    entropy_r = ((sample_count - i - 1.0f) / sample_count) * computeEntropyWithHist(histr, sample_count - i - 1, class_count);
    if(entropy_l + entropy_r < entropy_y) {
      entropy_y = entropy_l + entropy_r;
      splitPoint = (f_l[i].first + f_l[i + 1].first) / 2.0f;
    }
  }
  return {entropy - entropy_y, splitPoint};
}

signed main() {
  float *h_feature = (float *)malloc(dataSize * dataDim * sizeof(float));
  unsigned *h_label = (unsigned *)malloc(dataSize * sizeof(unsigned));
  
  printf("Done");
  // 生成随机数据 (0-1之间)
  for (unsigned i = 0, I = 0; i < dataSize; ++i, I += dataDim) 
    for (unsigned j = 0, J = I; j < dataDim; ++j, ++J) 
      h_feature[J] = (float)rand() / RAND_MAX;
  for (unsigned i = 0; i < dataSize; i++) h_label[i] = rand() % classCount;
  
  // 计算熵
  unsigned long long TimeBegin = clock(), TimeEnd;
  std::pair<float, float> IGp = informationGain(h_feature, h_label, dataSize, classCount);
  TimeEnd = clock();
  printf("IG: %f Split %f Time %llu us\n", IGp.first, IGp.second, TimeEnd - TimeBegin);
  

  return 0;
}