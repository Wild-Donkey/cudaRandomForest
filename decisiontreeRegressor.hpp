#pragma once

#include <vector>
#include "metrics.hpp"
#include <cstddef>

class regressorNode {
public:
  regressorNode();
  regressorNode(float* feature, float* label, unsigned sample_count, unsigned sample_dim,
    unsigned min_samples_split);
  ~regressorNode();
  void fit(float* feature, float* label, unsigned sample_count, unsigned sample_dim,
    unsigned min_samples_split);
  float predictOne(float* feature, unsigned* ori_feature_index = NULL);
private:
  regressorNode* left, * right;
  unsigned sample_count, sample_dim, min_samples_split = 5;
  unsigned best_feature;
  float splitPoint, ySum;
};

class decisionTreeRegressor {
public:
  decisionTreeRegressor();
  decisionTreeRegressor(unsigned sample_count, unsigned feature_count, unsigned feature_used,
    unsigned min_samples_split);
  decisionTreeRegressor(unsigned sample_count, unsigned feature_count, unsigned feature_used,
    unsigned min_samples_split, float* feature, float* label, unsigned* ori_feature_index);
  decisionTreeRegressor(unsigned sample_count, unsigned feature_count, unsigned min_samples_split,
    float* feature, float* label);
  ~decisionTreeRegressor();
  void Print();
  void fit(float* feature, float* label);
  void fit(float* feature, float* label, unsigned* ori_feature_index);
  void predict(float* feature, unsigned testSize, float* p_label);
private:
  unsigned sample_count, feature_count, feature_used, min_samples_split = 5;
  unsigned* ori_feature_index = NULL;
  regressorNode* root;
};