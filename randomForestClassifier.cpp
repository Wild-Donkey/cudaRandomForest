#include "bagging.hpp"
#include "dataReader.hpp"
#include "decisiontreeClassifier.hpp"
#include "randomForestClassifier.hpp"

randomForestClassifier::randomForestClassifier() {
  n_trees = 10;
  min_samples_split = 5;
}
randomForestClassifier::randomForestClassifier(unsigned class_count, unsigned min_samples_split, unsigned n_trees) :
  class_count(class_count), min_samples_split(min_samples_split), n_trees(n_trees) {
}
randomForestClassifier::~randomForestClassifier() {
  for (unsigned i = 0; i < n_trees; ++i) {
    trees[i].~decisionTreeClassifier();
  }
  trees.clear();
}
void randomForestClassifier::fit(float* features, unsigned* labels, unsigned sample_count_, unsigned feature_count_, unsigned class_count_) {
  sample_count = sample_count_;
  feature_count = feature_count_;
  class_count = class_count_;
  fit(features, labels);
}
void randomForestClassifier::fit(float* features, unsigned* labels) {
  if (!sample_per_tree) sample_per_tree = sample_count / 3;
  if (!feature_per_tree) feature_per_tree = sqrt(feature_count) + 1;

  void** dst_feature;
  void** dst_label;
  unsigned** ori_feature_index;
  randomSampling4Bytes(features, labels, dst_feature, dst_label, ori_feature_index,
    sample_count, feature_count, sample_per_tree, feature_per_tree, n_trees);

  for (unsigned i = 0; i < n_trees; ++i) {
    trees.push_back(decisionTreeClassifier(sample_per_tree, feature_per_tree, class_count, min_samples_split,
      ((float**)dst_feature)[i], ((unsigned**)dst_label)[i], ori_feature_index[i]));
    free(dst_feature[i]);
    free(dst_label[i]);
  }
  free(dst_feature);
  free(dst_label);
}
void randomForestClassifier::predict(float* features, unsigned* predictions, unsigned sample_count) {
  unsigned* pre_per_tree = (unsigned*)malloc(sample_count * n_trees * sizeof(unsigned));
  unsigned* Hist = (unsigned*)malloc(class_count * sizeof(unsigned));
  memset(predictions, 0, sample_count * sizeof(unsigned));
  for (unsigned i = 0, I = 0; i < n_trees; ++i, I += sample_count) {
    trees[i].predict(features, sample_count, pre_per_tree + I);
    for (unsigned j = 0; j < sample_count; ++j) predictions[j] += pre_per_tree[j];
  }
  for (unsigned i = 0; i < sample_count; ++i) {
    memset(Hist, 0, class_count * sizeof(unsigned));
    for (unsigned j = 0, J = i; j < n_trees; ++j, J += sample_count) Hist[pre_per_tree[J]]++;
    unsigned max = 0;
    for (unsigned j = 1; j < class_count; ++j)
      if (Hist[j] > Hist[max]) max = j;
    predictions[i] = max;
  }
  free(pre_per_tree);
}

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
  randomForestClassifier RFC(metaData->class_count, 5, 10);
  RFC.fit(train_feature, train_label);
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

  return 0;
}