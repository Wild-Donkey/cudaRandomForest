#include <vector>
class classifierNode {
  public:
    unsigned predictOne(std::vector<float> &sample);
  private:
    unsigned sample_count, vote_class, best_feature;
    std::vector<classifierNode*> children;
  
}
class decitionTreeClassifier {
  public:
    decitionTreeClassifier(std::vector<std::vector<float>> &samples, std::vector<unsigned> &labels);
    unsigned predictOne(std::vector<float> &sample);
    void train(std::vector<std::vector<float>> &samples, std::vector<unsigned> &labels);
    void test(std::vector<std::vector<float>> &samples, std::vector<unsigned> &labels);
    void printTree();
  private:
    classifierNode *root;
}