#pragma once

void computeHistogram(unsigned* label, unsigned*& hist, unsigned sample_count, unsigned class_count);
float computeEntropyWithHist(unsigned* hist, unsigned sample_count, unsigned class_count);
float computeEntropy(unsigned* label, unsigned sample_count, unsigned class_count);
float variance(float* label, unsigned sample_count);
float variance(unsigned size, float Sum, float Sum2);
std::pair<float, float> variance_sum(float* label, unsigned sample_count);
std::pair<float, float> informationGain(float* feature, unsigned* label, unsigned sample_count, unsigned class_count);
std::pair<unsigned, float> findBestFeature(float* feature, unsigned* label, unsigned sample_count,
  unsigned dataDim, unsigned class_count);
std::pair<unsigned, float> findBestFeature(float* feature, float* label, unsigned sample_count,
  unsigned dataDim);