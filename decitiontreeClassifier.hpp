#include <vector>
#include "metrics.hpp"

class classifierNode {
public:
  unsigned predictOne(float* feature);
  void fit(float* feature, unsigned* label, unsigned sample_count,
    unsigned sample_dim, unsigned class_count, unsigned min_samples_split);
private:
  unsigned sample_count, vote_class, best_feature;
  float splitPoint;
  classifierNode* left, * right;
};
class decitionTreeClassifier {
public:
  decitionTreeClassifier();
  ~decitionTreeClassifier();
  decitionTreeClassifier(std::vector<std::vector<float>>& samples, std::vector<unsigned>& labels);
  void fit(std::vector<std::vector<float>>& samples, std::vector<unsigned>& labels);
  unsigned predictOne(std::vector<float>& sample);
  unsigned predict(std::vector<float>& sample);
  void Print();
private:
  classifierNode* root;
  unsigned min_samples_split = 5;
  unsigned sample_count, vote_class, best_feature;
};
