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
  unsigned sample_count, unsigned sample_dim, unsigned class_count, unsigned min_samples_split) {

  vote_class = 0;
  for (unsigned i = 0; i < class_count; ++i)
    if (hist[i] > hist[vote_class]) vote_class = i;
  unsigned vote = hist[vote_class];
  // std::cout << "HistPost" << std::endl;
  free(hist);
  // std::cout << "Freed" << std::endl;

  // printf("Compute Histogram %u\n", vote);

  if (vote / sample_count > 0.90f || sample_count < min_samples_split) {
    left = right = NULL;
    return;
  }

  // std::cout << "herePreBest" << std::endl;
  auto best = findBestFeature(feature, label, sample_count, sample_dim, class_count);
  // std::cout << "herePostBest" << std::endl;
  best_feature = best.first;
  splitPoint = best.second;

  // std::cout << "here bested" << std::endl;
  // std::cout << "Cnt" << sample_count << "Fea" << sample_dim << std::endl;

  float* downFeature = (float*)malloc(sample_count * sample_dim * sizeof(float));
  unsigned* downLabel = (unsigned*)malloc(sample_count * sizeof(unsigned));
  // std::cout << "here Downed" << std::endl;

  float* rightFeature = downFeature + sample_count * sample_dim;
  float* leftFeature = downFeature;
  unsigned* rightLabel = downLabel + sample_count;
  unsigned* leftLabel = downLabel;

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
  // printf("Left %u Right %u\n", leftCount, rightCount);
  if (leftCount == 0 || rightCount == 0) {
    left = right = NULL;
    // printf("Free %p %p\n", downFeature, downLabel);
    // std::cout << "here" << std::endl;
    free(downFeature);
    free(downLabel);
    // std::cout << "Freed" << std::endl;
    return;
  }

  leftLabel -= leftCount;
  leftFeature -= leftCount * sample_dim;

  left = new(classifierNode);
  right = new(classifierNode);
  // printf("Node %p left %p right %p\n", this, left, right);
  left->fit(leftFeature, leftLabel, leftCount, sample_dim, class_count, min_samples_split);
  right->fit(rightFeature, rightLabel, rightCount, sample_dim, class_count, min_samples_split);
  // printf("Free %p %p\n", downFeature, downLabel);
  // std::cout << "PreFree" << std::endl;
  free(downFeature);
  free(downLabel);
  // std::cout << "Freed" << std::endl;
}

unsigned classifierNode::predictOne(float* feature, unsigned* ori_feature_index) {
  if (!left) return vote_class;
  unsigned BestIndex = ori_feature_index == NULL ? best_feature : ori_feature_index[best_feature];
  if (feature[BestIndex] <= splitPoint) {
    return left->predictOne(feature, ori_feature_index);
  } else return right->predictOne(feature, ori_feature_index);
}


decisionTreeClassifier::decisionTreeClassifier() { root = NULL; }
decisionTreeClassifier::decisionTreeClassifier(unsigned sample_count, unsigned feature_count, unsigned feature_used,
  unsigned class_count, unsigned min_samples_split) : sample_count(sample_count), feature_count(feature_count),
  feature_used(feature_used), class_count(class_count), min_samples_split(min_samples_split) {
  root = NULL;
}
decisionTreeClassifier::decisionTreeClassifier(unsigned sample_count, unsigned feature_count,
  unsigned class_count, unsigned min_samples_split, float* feature, unsigned* label) :
  sample_count(sample_count), feature_count(feature_count), feature_used(feature_used), class_count(class_count),
  min_samples_split(min_samples_split) {
  fit(feature, label);
}
decisionTreeClassifier::decisionTreeClassifier(unsigned sample_count, unsigned feature_count, unsigned feature_used,
  unsigned class_count, unsigned min_samples_split, float* feature, unsigned* label, unsigned* ori_feature_index) :
  sample_count(sample_count), feature_count(feature_count), feature_used(feature_used), class_count(class_count),
  min_samples_split(min_samples_split), ori_feature_index(ori_feature_index) {
  fit(feature, label);
}
decisionTreeClassifier::~decisionTreeClassifier() {
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
void decisionTreeClassifier::Print() {
  printf("decisionTreeClassifier: Root %p\n", root);
  printf("sample_count %u, feature_count %u, class_count %u\n", sample_count, feature_count, class_count);
}
void decisionTreeClassifier::fit(float* feature, unsigned* label) {
  printf("Fit\n");
  Print();
  root = new(classifierNode);
  root->fit(feature, label, sample_count, feature_used, class_count, min_samples_split);
  std::cout << "Fit Done" << std::endl;
}
void decisionTreeClassifier::predict(float* feature, unsigned testSize, unsigned* p_label) {
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