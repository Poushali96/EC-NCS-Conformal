"""Minimal stand-in for shap.TreeExplainer on sklearn forests (see /home/claude/tools/treeshap.py)."""
import sys
sys.path.insert(0, "/home/claude/tools")
from treeshap import ForestSHAP
__version__ = "local-cext"
class TreeExplainer:
    def __init__(self, model, feature_perturbation="tree_path_dependent", **kw):
        assert feature_perturbation == "tree_path_dependent"
        self._f = ForestSHAP(model)
    def shap_values(self, X, check_additivity=False, approximate=False, **kw):
        return self._f.shap_values(X, approximate=approximate)
