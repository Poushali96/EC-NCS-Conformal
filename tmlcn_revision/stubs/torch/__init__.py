"""Stub: the CICV5G rerun uses only the random-forest predictor."""
class _Cuda:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def empty_cache(): pass
    @staticmethod
    def manual_seed_all(s): pass
cuda = _Cuda()
class _Cudnn: deterministic = False; benchmark = False
class _Backends: cudnn = _Cudnn()
backends = _Backends()
def device(name): return name
def manual_seed(s): pass
float32 = "float32"
from . import nn
