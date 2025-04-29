#include <stdio.h>
#include <math.h>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <iostream>
#include "metrics.hpp"

void computeHistogram(unsigned* label, unsigned*& hist, unsigned sample_count, unsigned class_count) {
  hist = (unsigned*)malloc(class_count * sizeof(unsigned));
  memset(hist, 0, class_count * sizeof(unsigned));

  for (unsigned i = 0; i < sample_count; i++) ++hist[label[i]];
}
float computeEntropyWithHist(unsigned* hist, unsigned sample_count, unsigned class_count) {
  if (sample_count == 0) return 0.0f;
  float entropy = 0.0f;

  for (unsigned i = 0; i < class_count; i++) {
    float probab = (float)hist[i] / sample_count;
    entropy -= probab * log2(probab);
  }
  return entropy;
}
float computeEntropy(unsigned* label, unsigned sample_count, unsigned class_count) {
  if (sample_count == 0) return 0.0f;
  unsigned* hist;
  computeHistogram(label, hist, sample_count, class_count);
  float entropy = computeEntropyWithHist(hist, sample_count, class_count);
  free(hist);
  return entropy;
}
float variance(float* label, unsigned sample_count) {
  float Sum = 0, SqSum = 0;
  for (unsigned i = 0; i < sample_count; ++i) {
    Sum += label[i];
    SqSum += label[i] * label[i];
  }
  Sum /= sample_count;
  return SqSum / sample_count - Sum * Sum;
}
std::pair<float, float> variance_sum(float* label, unsigned sample_count) {
  float Sum = 0, SqSum = 0;
  for (unsigned i = 0; i < sample_count; ++i) {
    Sum += label[i];
    SqSum += label[i] * label[i];
  }
  Sum /= sample_count;
  return { SqSum / sample_count - Sum * Sum, Sum * sample_count };
}
float variance(unsigned size, float Sum, float Sum2) {
  Sum /= size;
  return Sum2 / size - Sum * Sum;
}
std::pair<float, float> informationGain(float* feature, unsigned* label, unsigned sample_count, unsigned class_count) {
  // std::cout << "Information Gain " << sample_count << std::endl;
  if (sample_count <= 1) return { 0.0f, 0.0f };

  float entropy, entropy_y, splitPoint;
  float entropy_l, entropy_r;
  std::pair<float, unsigned>* f_l = (std::pair<float, unsigned> *)malloc(sample_count * sizeof(std::pair<float, unsigned>));
  unsigned* histl = (unsigned*)malloc(class_count * sizeof(unsigned)), * histr;
  memset(histl, 0, class_count * sizeof(unsigned));
  computeHistogram(label, histr, sample_count, class_count);

  for (unsigned i = 0; i < sample_count; ++i) f_l[i] = { feature[i], label[i] };
  std::sort(f_l, f_l + sample_count);

  entropy_r = entropy = computeEntropyWithHist(histr, sample_count, class_count);
  entropy_l = 0;
  entropy_y = entropy_l + entropy_r;
  splitPoint = f_l[0].first - 0.001f;

  // std::cout << "L " << histl << " R " << histr << std::endl;

  // unsigned LeftCnt = 0, RightCnt = sample_count;
  // unsigned BL = 0, BR = RightCnt;

  for (unsigned i = 0; i < sample_count - 1; ++i) {
    histl[f_l[i].second]++;
    histr[f_l[i].second]--;

    // ++LeftCnt, --RightCnt;

    if (f_l[i].first == f_l[i + 1].first) continue;
    entropy_l = ((i + 1.0f) / sample_count) * computeEntropyWithHist(histl, i + 1, class_count);
    entropy_r = ((sample_count - i - 1.0f) / sample_count) * computeEntropyWithHist(histr, sample_count - i - 1, class_count);
    if (entropy_l + entropy_r < entropy_y) {
      entropy_y = entropy_l + entropy_r;
      splitPoint = (f_l[i].first + f_l[i + 1].first) / 2.0f;
      // BL = LeftCnt, BR = RightCnt;
    }
  }
  // std::cout << "Free " << f_l << " " << histl << " " << histr << std::endl;
  free(f_l);
  free(histl);
  free(histr);
  // printf("Best L %u R %u\n", BL, BR);
  return { entropy - entropy_y, splitPoint };
}
std::pair<unsigned, float> findBestFeature(float* feature, unsigned* label, unsigned sample_count,
  unsigned dataDim, unsigned class_count) {
  // std::cout << "Find Best Feature " << sample_count << " " << dataDim << std::endl;
  if (sample_count <= 1) return { 0, 0 };
  // std::cout << "Not emptyBest Feature " << std::endl;
  float maxIG = -0.01f;
  float splitPoint = 0.0f;
  unsigned bestFeature = 0;
  float* featureColumn = (float*)malloc(sample_count * sizeof(float));

  for (unsigned i = 0; i < dataDim; ++i) {
    for (unsigned j = 0, J = i; j < sample_count; ++j, J += dataDim)
      featureColumn[j] = feature[J];
    std::pair<float, float> IGp = informationGain(featureColumn, label, sample_count, class_count);
    // printf("Feature %u IG %f Split %f\n", I, IGp.first, IGp.second);
    if (IGp.first > maxIG) {
      maxIG = IGp.first;
      splitPoint = IGp.second;
      bestFeature = i;
    }
  }
  // std::cout << "Best Feature " << bestFeature << " Split " << splitPoint << std::endl;
  free(featureColumn);
  return { bestFeature, splitPoint };
}
std::pair<unsigned, float> findBestFeature(float* feature, float* label, unsigned sample_count,
  unsigned dataDim) {
  if (sample_count <= 1) return { 0, 0 };
  float minV = 1e30f;
  float splitPoint = 0.0f;
  unsigned bestFeature = 0;
  std::pair<float, float>* featureColumn = (std::pair<float, float>*)malloc(sample_count * sizeof(std::pair<float, float>));
  float* PreSum = (float*)malloc(sample_count * sizeof(float));
  float* PreSum2 = (float*)malloc(sample_count * sizeof(float));

  for (unsigned i = 0; i < dataDim; ++i) {
    for (unsigned j = 0, J = i; j < sample_count; ++j, J += dataDim)
      featureColumn[j] = { feature[J], label[j] };
    sort(featureColumn, featureColumn + sample_count);
    PreSum[0] = featureColumn[0].second;
    PreSum2[0] = featureColumn[0].second * featureColumn[0].second;
    for (unsigned j = 1; j < sample_count; ++j) {
      PreSum[j] = PreSum[j - 1] + featureColumn[j].second;
      PreSum2[j] = PreSum2[j - 1] + featureColumn[j].second * featureColumn[j].second;
    }
    float RSum = 0, RSum2 = 0, NewV;
    for (unsigned j = sample_count - 1; j; --j) {
      RSum += featureColumn[j].second;
      RSum2 += featureColumn[j].second * featureColumn[j].second;
      NewV = variance(j, PreSum[j], PreSum2[j]) + variance(sample_count - j, RSum, RSum2);
      if (NewV < minV) {
        minV = NewV;
        splitPoint = (featureColumn[j].first + featureColumn[j - 1].first) / 2.0f;
        bestFeature = i;
      }
    }
  }
  // std::cout << "Best Feature " << bestFeature << " Split " << splitPoint << std::endl;
  free(featureColumn);
  free(PreSum);
  free(PreSum2);
  return { bestFeature, splitPoint };
}

