#include "decisiontreeRegressor.hpp"
#include "metrics.hpp" 
#include "dataReader.hpp"

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

  this->sample_count = sample_count;
  this->min_samples_split = min_samples_split;
  this->sample_dim = sample_dim;
  auto Var_Sum = variance_sum(label, sample_count);
  ySum = Var_Sum.second;
  float Var = Var_Sum.first;
  // printf("Var %f Mean %f n %u min %u\n", Var, ySum / sample_count, sample_count, min_samples_split);
  if (Var <= ySum * 0.001 / sample_count || sample_count < min_samples_split) {
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
  // printf("Predict One %u %u %f\n", sample_count, min_samples_split, ySum / sample_count);
  if (!left) return ySum / sample_count;
  unsigned BestIndex = ori_feature_index == NULL ? best_feature : ori_feature_index[best_feature];
  // std::cout << "bestIndex " << BestIndex << " splitPoint " << splitPoint << std::endl;
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
  sample_count(sample_count), feature_count(feature_count),
  min_samples_split(min_samples_split) {
  feature_used = feature_count;
  // std::cout << "sample_count " << sample_count << " feature_count " << feature_count << std::endl;
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
  printf("decisionTreeRegressor: Root %p\n", root);
  printf("sample_count %u, feature_count %u\n", sample_count, feature_count);
}
void decisionTreeRegressor::fit(float* feature, float* label) {
  // std::cout << "Fit" << std::endl;
  root = new(regressorNode);
  root->fit(feature, label, sample_count, feature_used, min_samples_split);
  // Print();
  // std::cout << "Fit Done" << std::endl;
}
void decisionTreeRegressor::predict(float* feature, unsigned testSize, float* p_label) {
  for (unsigned i = 0, I = 0; i < testSize; ++i, I += feature_count) {
    p_label[i] = root->predictOne(feature + I, ori_feature_index);
  }
}
signed main() {
  metaData* metaData;
  read_config(metaData, "data/adult.myc");
  // std::cout << "Samples " << metaData->sample_count << " Features " << metaData->feature_count << std::endl;
  float* h_data = (float*)malloc(metaData->sample_count * metaData->feature_count * sizeof(float));
  bool* is_category_feature = (bool*)malloc(metaData->feature_count * sizeof(bool));
  get_data(h_data, is_category_feature, *metaData, read_csv("data/adult.csv"));

  unsigned trainSize(metaData->sample_count * 0.8), testSize(metaData->sample_count - trainSize);

  float* train_data = h_data;
  float* test_data = h_data + (trainSize * metaData->feature_count);

  for (unsigned i = 0; i < metaData->feature_count; ++i) if (!is_category_feature[i]) {
    float* trainX = (float*)malloc(trainSize * (metaData->feature_count - 1) * sizeof(float));
    float* trainY = (float*)malloc(trainSize * sizeof(float));
    for (unsigned j = 0, J = 0, JJ = 0; j < trainSize; ++j, J += metaData->feature_count - 1, JJ += metaData->feature_count) {
      if (i) memcpy(trainX + J, train_data + JJ, i);
      if (metaData->feature_count - 1 != i) memcpy(trainX + J + i, train_data + JJ + i + 1, metaData->feature_count - i - 1);
      trainY[j] = train_data[JJ + i];
    }
    // std::cout << "Samples " << metaData->sample_count << " Features " << metaData->feature_count << std::endl;
    unsigned long long TimeBegin = clock(), TimeEnd;
    decisionTreeRegressor dtc(trainSize, metaData->feature_count - 1, 2, trainX, trainY);
    TimeEnd = clock();
    printf("fit %u Time %f ms\n", trainSize, (TimeEnd - TimeBegin) / 1000.0f);


    float* testX = (float*)malloc(testSize * (metaData->feature_count - 1) * sizeof(float));
    float* testY = (float*)malloc(testSize * sizeof(float));
    float* test_predict = (float*)malloc(testSize * sizeof(float));

    for (unsigned j = 0, J = 0, JJ = 0; j < testSize; ++j, J += metaData->feature_count - 1, JJ += metaData->feature_count) {
      if (i) memcpy(testX + J, test_data + JJ, i);
      if (metaData->feature_count - 1 != i) memcpy(testX + J + i, test_data + JJ + i + 1, metaData->feature_count - i - 1);
      testY[j] = test_data[JJ + i];
    }

    TimeBegin = clock();
    dtc.predict(testX, testSize, test_predict);
    TimeEnd = clock();
    printf("predict %u Time %f ms\n", testSize, (TimeEnd - TimeBegin) / 1000.0f);
    for (unsigned i = 0; i < 20; ++i)
      printf("%f ", test_predict[i]);
    putchar(0x0A);
    for (unsigned i = 0; i < 20; ++i)
      printf("%f ", testY[i]);
    putchar(0x0A);

    float Loss = 0;
    for (unsigned i = 0; i < testSize; ++i) {
      float Delt = testY[i] - test_predict[i];
      Loss += Delt * Delt;
    }
    printf("MSE %f\n", Loss / testSize);
    free(trainX);
    free(trainY);
    free(testX);
    free(testY);
    free(test_predict);
    // break;
  }
  free(h_data);
  free(is_category_feature);
  delete metaData;
  return 0;
}