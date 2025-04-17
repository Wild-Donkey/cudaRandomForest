#include "decitiontreeClassifier.hpp"
#include "metrics.hpp" 
#include <stdio.h>
#include <malloc.h>
#include <cstring>

void classifierNode::fit(float* feature, unsigned* label, unsigned sample_count,
  unsigned sample_dim, unsigned class_count, unsigned min_samples_split) {

  if (sample_count < min_samples_split) {
    unsigned* hist;
    computeHistogram(label, hist, sample_count, class_count);
    vote_class = 0;
    for (unsigned i = 0; i < class_count; ++i)
      if (hist[i] > hist[vote_class]) vote_class = i;
    left = right = NULL;
    return;
  }
  auto best = findBestFeature(feature, label, sample_count, sample_dim, class_count);
  best_feature = best.first;
  splitPoint = best.second;

  float* downFeature = (float*)malloc(sample_count * sample_dim * sizeof(float));
  unsigned* downLabel = (unsigned*)malloc(sample_count * sizeof(unsigned));
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
  unsigned rightCount = rightLabel - downLabel;
  leftLabel -= leftCount;
  leftFeature -= leftCount * sample_dim;

  left = new(classifierNode);
  right = new(classifierNode);
  left->fit(leftFeature, leftLabel, leftCount, sample_dim, class_count, min_samples_split);
  right->fit(leftFeature, leftLabel, leftCount, sample_dim, class_count, min_samples_split);
}
unsigned classifierNode::predictOne(float* feature) {

}

decitionTreeClassifier::decitionTreeClassifier() {
  root = NULL;
}
decitionTreeClassifier::~decitionTreeClassifier() {
  if (root) {
    delete root;
    root = NULL;
  }
}
void decitionTreeClassifier::Print() {
  printf("decitionTreeClassifier: Root %p", root);
}
void fit(float* x, unsigned* y, unsigned sample_count, unsigned class_count) {
  unsigned sample_dim = 0;
  unsigned min_samples_split = 2;
  root = new(classifierNode);
  root->fit(x, y, sample_count, sample_dim, class_count, min_samples_split);
}
void predict(float* x, unsigned sample_count, unsigned* y);

signed main() {
  decitionTreeClassifier dtc;

}