#include <Python.h>
#include "tree_shap.h"
extern "C" void treeshap_path_dependent(
    int *children_left, int *children_right, int *children_default, int *features,
    double *thresholds, int *threshold_types, double *values, double *node_sample_weights,
    unsigned max_depth, unsigned num_trees, double *base_offset, unsigned max_nodes,
    unsigned num_outputs, double *X, bool *X_missing, unsigned num_X, unsigned M,
    double *out_contribs) {
    TreeEnsemble trees(children_left, children_right, children_default, features, thresholds,
                       threshold_types, values, node_sample_weights, max_depth, num_trees,
                       base_offset, max_nodes, num_outputs);
    ExplanationDataset data(X, X_missing, nullptr, nullptr, nullptr, num_X, M, 0);
    dense_tree_shap(trees, data, out_contribs, FEATURE_DEPENDENCE::tree_path_dependent,
                    MODEL_TRANSFORM::identity, false);
}
extern "C" void treeshap_saabas(
    int *children_left, int *children_right, int *children_default, int *features,
    double *thresholds, int *threshold_types, double *values, double *node_sample_weights,
    unsigned max_depth, unsigned num_trees, double *base_offset, unsigned max_nodes,
    unsigned num_outputs, double *X, bool *X_missing, unsigned num_X, unsigned M,
    double *out_contribs) {
    TreeEnsemble trees(children_left, children_right, children_default, features, thresholds,
                       threshold_types, values, node_sample_weights, max_depth, num_trees,
                       base_offset, max_nodes, num_outputs);
    ExplanationDataset data(X, X_missing, nullptr, nullptr, nullptr, num_X, M, 0);
    dense_tree_saabas(out_contribs, trees, data);
}
