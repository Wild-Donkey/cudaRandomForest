#pragma once

#include <vector>
#include "decisiontreeRegressor.hpp"


class randomForestRegressor {
public:
  randomForestRegressor();
  randomForestRegressor(unsigned min_samples_split, unsigned n_trees);
  ~randomForestRegressor();
  void fit(float* features, float* labels);
  void fit(float* features, float* labels, unsigned sample_count_, unsigned feature_count_);
  void predict(float* features, float* labels, unsigned sample_count);
private:
  decisionTreeRegressor* trees;
  unsigned n_trees;
  unsigned sample_count, feature_count;
  unsigned sample_per_tree = 0, feature_per_tree = 0;
  unsigned min_samples_split;
};