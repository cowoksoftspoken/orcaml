from .orca_python import Tensor, DType, Device, save_tensors, load_tensors, set_grad_enabled, is_grad_enabled
from .tensor import einsum
from .autocast import autocast, GradScaler
from . import nn
from . import optim
from . import data
from . import callbacks
from . import onnx
from . import zoo
from . import hf
from . import distributed

# Factory functions
zeros = Tensor.zeros
ones = Tensor.ones
scalar = Tensor.scalar
randn = Tensor.randn
rand_uniform = Tensor.rand_uniform
rand_dropout_mask = Tensor.rand_dropout_mask
from_list = Tensor.from_list

class no_grad:
    """Context manager and decorator to temporarily disable gradient computation."""
    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_grad_enabled(self.prev)

    def __call__(self, func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

__version__ = "0.5.0"
__all__ = [
    "Tensor", "DType", "Device", "save_tensors", "load_tensors", "einsum", 
    "autocast", "GradScaler", "nn", "optim", "data", "callbacks", "onnx", "zoo", "hf",
    "distributed", "zeros", "ones", "scalar", "randn", "rand_uniform",
    "rand_dropout_mask", "from_list", "no_grad", "set_grad_enabled", "is_grad_enabled"
]
