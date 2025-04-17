#include<dicitiontree.hpp>

class decitionTreeClassifier {
public:
    decitionTreeClassifier();
    ~decitionTreeClassifier();
    void fit(float* x, unsigned* y, unsigned sample_count, unsigned class_count);
    void predict(float* x, unsigned sample_count, unsigned* y);
    void saveModel(const char* filename);
    void loadModel(const char* filename);
private:
    unsigned* h_label;
    unsigned* h_feature;
    unsigned* h_sample_count;
    unsigned* h_class_count;
    unsigned* h_feature_count;
    unsigned* h_tree_depth;
    unsigned* h_max_depth;
    unsigned* h_min_samples_split;
    unsigned* h_min_samples_leaf;
    unsigned* h_max_features;
    unsigned* h_criterion;
    unsigned* h_splitter;
    unsigned* h_random_state;
    unsigned* h_n_jobs;
}