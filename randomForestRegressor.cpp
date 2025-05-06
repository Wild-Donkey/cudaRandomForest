#include "bagging.hpp"
#include "dataReader.hpp"
#include "decisiontreeRegressor.hpp"
#include "randomForestRegressor.hpp"

#include <algorithm>

randomForestRegressor::randomForestRegressor() {
  n_trees = 10;
  min_samples_split = 5;
}
randomForestRegressor::randomForestRegressor(unsigned min_samples_split, unsigned n_trees) :
  n_trees(n_trees), min_samples_split(min_samples_split) {
}
randomForestRegressor::~randomForestRegressor() {
  for (unsigned i = 0; i < n_trees; ++i) {
    trees[i].~decisionTreeRegressor();
  }
  free(trees);
}
void randomForestRegressor::fit(float* features, float* labels, unsigned sample_count_,
  unsigned feature_count_) {
  sample_count = sample_count_;
  feature_count = feature_count_;
  fit(features, labels);
}
void randomForestRegressor::fit(float* features, float* labels) {
  if (!sample_per_tree) sample_per_tree = sample_count / 3;
  if (!feature_per_tree) feature_per_tree = feature_count / 3;

  void** dst_feature;
  void** dst_label;
  unsigned** ori_feature_index;
  randomSampling4Bytes(features, labels, dst_feature, dst_label, ori_feature_index,
    sample_count, feature_count, sample_per_tree, feature_per_tree, n_trees);

  // printf("Bagged %u for %u * %u\n", sample_count, sample_per_tree, n_trees);

  std::cout << "SamplePerTree " << sample_per_tree << " featurePreTree " << feature_per_tree << std::endl;
  trees = (decisionTreeRegressor*)malloc(n_trees * sizeof(decisionTreeRegressor));
  for (unsigned i = 0; i < n_trees; ++i) {
    new (&trees[i]) decisionTreeRegressor(
      sample_per_tree, feature_count, feature_per_tree,
      min_samples_split,
      (float*)dst_feature[i], (float*)dst_label[i], ori_feature_index[i]
    );
    // std::cout << "deCfit " << i << std::endl;
    free(dst_feature[i]);
    free(dst_label[i]);
  }
  free(dst_feature);
  free(dst_label);
  free(ori_feature_index);
}
void randomForestRegressor::predict(float* features, float* predictions, unsigned sample_count) {
  float* pre_per_tree = (float*)malloc(sample_count * n_trees * sizeof(float));
  memset(predictions, 0, sample_count * sizeof(float));
  for (unsigned i = 0, I = 0; i < n_trees; ++i, I += sample_count) {
    trees[i].predict(features, sample_count, pre_per_tree + I);
    for (unsigned j = 0; j < sample_count; ++j) predictions[j] += pre_per_tree[j];
  }
  unsigned L = n_trees * 0.2, R = n_trees * 0.8;
  // printf("n_trees %u L %u R %u\n", n_trees, L, R);
  for (unsigned i = 0; i < sample_count; ++i) {
    std::sort(pre_per_tree + i * n_trees, pre_per_tree + (i + 1) * n_trees);
    float Sum = 0;
    for (unsigned j = L; j < R; ++j) Sum += pre_per_tree[i * n_trees + j];
    predictions[i] = Sum / (R - L);
  }
  free(pre_per_tree);
}
/*
signed main() {
  metaData* metaData;
  read_config(metaData, "data/adult.myc");

  float* h_feature = (float*)malloc(metaData->sample_count * metaData->feature_count * sizeof(float));
  unsigned* h_label = (unsigned*)malloc(metaData->sample_count * sizeof(unsigned));
  getfeat_and_label(h_feature, h_label, *metaData, read_csv("data/adult.csv"));

  unsigned trainSize(metaData->sample_count * 0.8), testSize(metaData->sample_count - trainSize);
  float* train_feature = h_feature;
  unsigned* train_label = h_label;

  unsigned long long TimeBegin = clock(), TimeEnd;
  randomForestRegressor RFC(metaData->class_count, 5, 10);
  printf("Hello\n");
  RFC.fit(train_feature, train_label, trainSize, metaData->feature_count, metaData->class_count);
  TimeEnd = clock();
  printf("fit %u Time %f ms\n", trainSize, (TimeEnd - TimeBegin) / 1000.0f);

  float* test_feature = h_feature + trainSize * metaData->feature_count;
  unsigned* test_label = h_label + trainSize;
  unsigned* test_predict = (unsigned*)malloc(testSize * sizeof(unsigned));

  TimeBegin = clock();
  RFC.predict(test_feature, test_predict, testSize);
  TimeEnd = clock();
  printf("predict %u Time %f ms\n", testSize, (TimeEnd - TimeBegin) / 1000.0f);

  unsigned Loss = 0;
  for (unsigned i = 0; i < testSize; ++i) {
    if (test_label[i] != test_predict[i]) ++Loss;
  }
  printf("Loss %f\n", (float)Loss / testSize);
  delete metaData;
  free(h_feature);
  free(h_label);
  free(test_predict);
  return 0;
}
*/