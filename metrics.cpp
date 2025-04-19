#include <stdio.h>
#include <math.h>
#include <ctime>
#include <cstring>
#include <algorithm>
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

std::pair<float, float> informationGain(float* feature, unsigned* label, unsigned sample_count, unsigned class_count) {
  if (sample_count == 0) return { 0.0f, 0.0f };

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
  // printf("Best L %u R %u\n", BL, BR);
  return { entropy - entropy_y, splitPoint };
}
std::pair<unsigned, float> findBestFeature(float* feature, unsigned* label,
  unsigned sample_count, unsigned dataDim, unsigned class_count) {
  if (sample_count == 0) return { 0, 0 };

  float maxIG = -0.01f;
  float splitPoint = 0.0f;
  unsigned bestFeature = 0;
  float* featureColumn = (float*)malloc(sample_count * sizeof(float));

  for (unsigned i = 0; i < dataDim; ++i) {
    for (unsigned j = 0, J = i; j < sample_count; ++j, J += dataDim)
      featureColumn[j] = feature[J];
    std::pair<float, float> IGp = informationGain(featureColumn, label, sample_count, class_count);
    // printf("Feature %u IG %f Split %f\n", i, IGp.first, IGp.second);
    if (IGp.first > maxIG) {
      maxIG = IGp.first;
      splitPoint = IGp.second;
      bestFeature = i;
    }
  }
  return { bestFeature, splitPoint };
}

