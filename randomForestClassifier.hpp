#pragma once

#include <vector>
#include "decisiontreeClassifier.hpp"


class randomForestClassifier {
public:
  randomForestClassifier();
  randomForestClassifier(unsigned class_count, unsigned min_samples_split, unsigned n_trees);
  ~randomForestClassifier();
  void fit(float* features, unsigned* labels);
  void fit(float* features, unsigned* labels, unsigned sample_count_, unsigned feature_count_, unsigned class_count_);
  void predict(float* features, unsigned* labels, unsigned sample_count);
private:
  decisionTreeClassifier* trees;
  unsigned n_trees = 10;
  unsigned sample_count, feature_count, class_count;
  unsigned sample_per_tree = 0, feature_per_tree = 0;
  unsigned min_samples_split = 5;
};