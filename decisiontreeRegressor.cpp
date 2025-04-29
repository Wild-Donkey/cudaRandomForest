#include "decisiontreeRegressor.hpp"
#include "metrics.hpp" 
// #include "dataReader.hpp"

#include <stdio.h>
#include <malloc.h>
#include <cstring>
#include <iostream>

regressorNode::regressorNode() { left = right = NULL; }
regressorNode::regressorNode(float* feature, float* label, unsigned sample_count,
  unsigned sample_dim, unsigned min_samples_split) {
  left = right = NULL;
  fit(feature, label, sample_count, sample_dim, min_samples_split);
}
regressorNode::~regressorNode() {
  if (left) delete left, left = NULL;
  if (right) delete right, right = NULL;
}
void regressorNode::fit(float* feature, float* label,
  unsigned sample_count, unsigned sample_dim, unsigned min_samples_split) {

  auto Var_Sum = variance_sum(label, sample_count);
  ySum = Var_Sum.second;
  float Var = Var_Sum.first;

  if (Var <= ySum * 0.01 / sample_count || sample_count < min_samples_split) {
    left = right = NULL;
    return;
  }

  auto best = findBestFeature(feature, label, sample_count, sample_dim);
  best_feature = best.first;
  splitPoint = best.second;

  float* downFeature = (float*)malloc(sample_count * sample_dim * sizeof(float));
  float* downLabel = (float*)malloc(sample_count * sizeof(float));

  float* rightFeature = downFeature + sample_count * sample_dim;
  float* leftFeature = downFeature;
  float* rightLabel = downLabel + sample_count;
  float* leftLabel = downLabel;

  for (unsigned i = 0, I = 0, Id = best_feature; i < sample_count; ++i, I += sample_dim, Id += sample_dim) {
    if (feature[Id] <= splitPoint) {
      memcpy(leftFeature, feature + I, sample_dim * sizeof(float));
      leftFeature += sample_dim;
      (*leftLabel) = label[i], ++leftLabel;
    } else {
      rightFeature -= sample_dim;
      memcpy(rightFeature, feature + I, sample_dim * sizeof(float));
      --rightLabel, (*rightLabel) = label[i];
    }
  }

  unsigned leftCount = leftLabel - downLabel;
  unsigned rightCount = sample_count - leftCount;
  if (leftCount == 0 || rightCount == 0) {
    left = right = NULL;
    free(downFeature);
    free(downLabel);
    return;
  }

  leftLabel -= leftCount;
  leftFeature -= leftCount * sample_dim;

  left = new(regressorNode);
  right = new(regressorNode);
  left->fit(leftFeature, leftLabel, leftCount, sample_dim, min_samples_split);
  right->fit(rightFeature, rightLabel, rightCount, sample_dim, min_samples_split);
  free(downFeature);
  free(downLabel);
}

float regressorNode::predictOne(float* feature, unsigned* ori_feature_index) {
  if (!left) return ySum / sample_count;
  unsigned BestIndex = ori_feature_index == NULL ? best_feature : ori_feature_index[best_feature];
  if (feature[BestIndex] <= splitPoint) {
    return left->predictOne(feature, ori_feature_index);
  } else return right->predictOne(feature, ori_feature_index);
}


decisionTreeRegressor::decisionTreeRegressor() { root = NULL; }
decisionTreeRegressor::decisionTreeRegressor(unsigned sample_count, unsigned feature_count, unsigned feature_used,
  unsigned min_samples_split) : sample_count(sample_count), feature_count(feature_count),
  feature_used(feature_used), min_samples_split(min_samples_split) {
  root = NULL;
}
decisionTreeRegressor::decisionTreeRegressor(unsigned sample_count, unsigned feature_count,
  unsigned min_samples_split, float* feature, float* label) :
  sample_count(sample_count), feature_count(feature_count), feature_used(feature_used),
  min_samples_split(min_samples_split) {
  fit(feature, label);
}
decisionTreeRegressor::decisionTreeRegressor(unsigned sample_count, unsigned feature_count, unsigned feature_used,
  unsigned min_samples_split, float* feature, float* label, unsigned* ori_feature_index) :
  sample_count(sample_count), feature_count(feature_count), feature_used(feature_used),
  min_samples_split(min_samples_split), ori_feature_index(ori_feature_index) {
  fit(feature, label);
}
decisionTreeRegressor::~decisionTreeRegressor() {
  // printf("Over For this Tree\n");
  if (root) {
    delete root;
    root = NULL;
  }
  if (ori_feature_index) {
    free(ori_feature_index);
    ori_feature_index = NULL;
  }
}
void decisionTreeRegressor::Print() {
  printf("decisionTreeClassifier: Root %p\n", root);
  printf("sample_count %u, feature_count %u, class_count %u\n", sample_count, feature_count);
}
void decisionTreeRegressor::fit(float* feature, float* label) {
  printf("Fit\n");
  Print();
  root = new(regressorNode);
  root->fit(feature, label, sample_count, feature_used, min_samples_split);
  std::cout << "Fit Done" << std::endl;
}
void decisionTreeRegressor::predict(float* feature, unsigned testSize, float* p_label) {
  for (unsigned i = 0, I = 0; i < testSize; ++i, I += feature_count) {
    p_label[i] = root->predictOne(feature + I, ori_feature_index);
  }
}
/*
signed main() {
  metaData* metaData;
  read_config(metaData, "data/adult.myc");

  float* h_feature = (float*)malloc(metaData->sample_count * metaData->feature_count * sizeof(float));
  unsigned* h_label = (unsigned*)malloc(metaData->sample_count * sizeof(unsigned));
  getfeat_and_label(h_feature, h_label, *metaData, read_csv("data/adult.csv"));

  unsigned trainSize(metaData->sample_count * 0.8), testSize(metaData->sample_count - trainSize);

  unsigned long long TimeBegin = clock(), TimeEnd;
  decisionTreeClassifier dtc(trainSize, metaData->feature_count,
    metaData->features_meta[metaData->feature_count].stringSet.size(), 5, h_feature, h_label);
  TimeEnd = clock();
  printf("fit %u Time %f ms\n", trainSize, (TimeEnd - TimeBegin) / 1000.0f);

  float* test_feature = h_feature + trainSize * metaData->feature_count;
  unsigned* test_label = h_label + trainSize;
  unsigned* test_predict;

  TimeBegin = clock();
  dtc.predict(test_feature, testSize, test_predict);
  TimeEnd = clock();
  printf("predict %u Time %f ms\n", testSize, (TimeEnd - TimeBegin) / 1000.0f);

  unsigned Loss = 0;
  for (unsigned i = 0; i < testSize; ++i) {
    if (test_label[i] != test_predict[i]) ++Loss;
  }
  printf("Loss %f\n", (float)Loss / testSize);

  return 0;
}
  */