"""Exact path-dependent TreeSHAP for scikit-learn regression forests.

Calls the TreeSHAP C++ routine from the official shap repository (compiled into
libtreeshap.so), using the same tree conversion as shap.TreeExplainer for
sklearn RandomForestRegressor / ExtraTreesRegressor.
"""
import ctypes
import os

import numpy as np

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtreeshap.so"))
_P_INT = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
_P_DBL = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
_P_BOOL = np.ctypeslib.ndpointer(np.bool_, flags="C_CONTIGUOUS")
_lib.treeshap_path_dependent.argtypes = [
    _P_INT, _P_INT, _P_INT, _P_INT, _P_DBL, _P_INT, _P_DBL, _P_DBL,
    ctypes.c_uint, ctypes.c_uint, _P_DBL, ctypes.c_uint, ctypes.c_uint,
    _P_DBL, _P_BOOL, ctypes.c_uint, ctypes.c_uint, _P_DBL,
]
_lib.treeshap_path_dependent.restype = None
_lib.treeshap_saabas.argtypes = _lib.treeshap_path_dependent.argtypes
_lib.treeshap_saabas.restype = None


class ForestSHAP:
    def __init__(self, forest):
        trees = [e.tree_ for e in forest.estimators_]
        n_trees = len(trees)
        max_nodes = max(t.node_count for t in trees)
        cl = np.full((n_trees, max_nodes), -1, np.int32)
        cr = np.full((n_trees, max_nodes), -1, np.int32)
        feat = np.full((n_trees, max_nodes), -1, np.int32)
        thr = np.zeros((n_trees, max_nodes))
        val = np.zeros((n_trees, max_nodes, 1))
        w = np.zeros((n_trees, max_nodes))
        depth = 0
        for i, t in enumerate(trees):
            n = t.node_count
            cl[i, :n] = t.children_left
            cr[i, :n] = t.children_right
            feat[i, :n] = np.where(t.feature >= 0, t.feature, -2)  # match shap
            thr[i, :n] = t.threshold
            val[i, :n, 0] = t.value[:, 0, 0] / n_trees  # forest = mean of trees
            w[i, :n] = t.weighted_n_node_samples
            depth = max(depth, t.max_depth)
        self.cl, self.cr, self.cd = cl, cr, cl.copy()
        self.feat, self.thr = feat, thr
        self.ttype = np.zeros((n_trees, max_nodes), np.int32)
        self.val, self.w = val, w
        self.depth, self.n_trees, self.max_nodes = depth, n_trees, max_nodes
        self.base = np.zeros(1)
        self.n_features = forest.n_features_in_

    def shap_values(self, X, batch=20000, approximate=False):
        fn = _lib.treeshap_saabas if approximate else _lib.treeshap_path_dependent
        X = np.ascontiguousarray(X, np.float64)
        n, M = X.shape
        out = np.empty((n, M))
        self.expected = None
        for s in range(0, n, batch):
            xb = np.ascontiguousarray(X[s:s + batch])
            nb = len(xb)
            miss = np.ascontiguousarray(np.isnan(xb))
            contrib = np.zeros((nb, M + 1, 1))
            fn(
                self.cl, self.cr, self.cd, self.feat, self.thr, self.ttype,
                self.val, self.w, self.depth, self.n_trees, self.base,
                self.max_nodes, 1, xb, miss, nb, M, contrib.reshape(-1))
            out[s:s + nb] = contrib[:, :M, 0]
            self.expected = contrib[0, M, 0]
        return out
