#pragma once

#include <vector>
#include "metrics.hpp"
#include <cstddef>

class classifierNode {
public:
  classifierNode();
  classifierNode(float* feature, unsigned* label, unsigned sample_count, unsigned sample_dim,
    unsigned class_count, unsigned min_samples_split);
  ~classifierNode();
  void fit(float* feature, unsigned* label, unsigned sample_count, unsigned sample_dim,
    unsigned class_count, unsigned min_samples_split);
  unsigned predictOne(float* feature, unsigned* ori_feature_index = NULL);
private:
  classifierNode* left, * right;
  unsigned sample_count, sample_dim, class_count, min_samples_split = 5;
  unsigned vote_class, best_feature;
  float splitPoint;
};

class decisionTreeClassifier {
public:
  decisionTreeClassifier();
  decisionTreeClassifier(unsigned sample_count, unsigned feature_count, unsigned feature_used,
    unsigned class_count, unsigned min_samples_split);
  decisionTreeClassifier(unsigned sample_count, unsigned feature_count, unsigned feature_used,
    unsigned class_count, unsigned min_samples_split, float* feature, unsigned* label, unsigned* ori_feature_index);
  decisionTreeClassifier(unsigned sample_count, unsigned feature_count, unsigned class_count,
    unsigned min_samples_split, float* feature, unsigned* label);
  ~decisionTreeClassifier();
  void Print();
  void fit(float* feature, unsigned* label);
  void fit(float* feature, unsigned* label, unsigned* ori_feature_index);
  void predict(float* feature, unsigned testSize, unsigned* p_label);
private:
  unsigned sample_count, feature_count, feature_used, class_count, min_samples_split = 5;
  unsigned* ori_feature_index = NULL;
  classifierNode* root;
};