#pragma once

void computeHistogram(unsigned* label, unsigned*& hist, unsigned sample_count, unsigned class_count);
float computeEntropyWithHist(unsigned* hist, unsigned sample_count, unsigned class_count);
float computeEntropy(unsigned* label, unsigned sample_count, unsigned class_count);
std::pair<float, float> informationGain(float* feature, unsigned* label, unsigned sample_count, unsigned class_count);
std::pair<unsigned, float> findBestFeature(float* feature, unsigned* label, unsigned sample_count,
  unsigned dataDim, unsigned class_count);