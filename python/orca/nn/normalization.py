import orca
from .module import Module
from .parameter import Parameter
from orca.tensor import Tensor
from .activation import _scalar

class LayerNorm(Module):
    """
    Applies Layer Normalization over a mini-batch of inputs as described in the paper
    Layer Normalization.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.weight = Parameter(Tensor.ones([1, normalized_shape], requires_grad=True, device=device))
        self.bias = Parameter(Tensor.zeros([1, normalized_shape], requires_grad=True, device=device))

    def forward(self, x: Tensor) -> Tensor:
        shape = list(x.shape)
        if shape[-1] != self.normalized_shape:
            raise ValueError(f"Expected last dim {self.normalized_shape}, got {shape[-1]}")
            
        reduced_shape = list(shape)
        reduced_shape[-1] = 1
        
        F = float(self.normalized_shape)
        
        sum_x = x.sum_to_shape(reduced_shape)
        mean = sum_x * (1.0 / F)
        mean_exp = mean.expand(shape)
        
        diff = x - mean_exp
        squared = diff * diff
        
        sum_sq = squared.sum_to_shape(reduced_shape)
        var = sum_sq * (1.0 / F)
        
        eps_tensor = _scalar(self.eps, x.device).expand(reduced_shape)
        var_eps = var + eps_tensor
        std = var_eps.sqrt().expand(shape)
        
        norm = diff / std
        
        weight_exp = self.weight.tensor.expand(shape)
        bias_exp = self.bias.tensor.expand(shape)
        return (norm * weight_exp) + bias_exp


class BatchNorm2d(Module):
    """
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension).
    This performs proper 4D batch normalization across the spatial and batch dimensions.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Parameter(Tensor.ones([1, num_features, 1, 1], requires_grad=True, device=device))
        self.bias = Parameter(Tensor.zeros([1, num_features, 1, 1], requires_grad=True, device=device))
        
        # Buffers for running statistics
        self.register_buffer('running_mean', Tensor.zeros([1, num_features, 1, 1], requires_grad=False, device=device))
        self.register_buffer('running_var', Tensor.ones([1, num_features, 1, 1], requires_grad=False, device=device))

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        if len(shape) != 4:
            raise ValueError(f"BatchNorm2d expected 4D input, got {len(shape)}D")
            
        if self.training:
            # Compute mean and variance across N, H, W
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            num_elements = N * H * W
            
            reduced_shape = [1, C, 1, 1]
            
            sum_x = x.sum_to_shape(reduced_shape)
            mean = sum_x * (1.0 / num_elements)
            mean_exp = mean.expand(shape)
            
            diff = x - mean_exp
            squared = diff * diff
            sum_sq = squared.sum_to_shape(reduced_shape)
            
            # Bessel's correction for unbiased variance if N*H*W > 1
            unbiased_denom = max(1.0, float(num_elements - 1))
            var = sum_sq * (1.0 / num_elements)
            unbiased_var = sum_sq * (1.0 / unbiased_denom)
            
            # Update running stats (momentum) using zero-copy detach
            old_rm = self._buffers['running_mean']
            old_rv = self._buffers['running_var']
            
            new_rm = (old_rm * (1.0 - self.momentum)) + (mean * self.momentum)
            new_rv = (old_rv * (1.0 - self.momentum)) + (unbiased_var * self.momentum)
            
            self._buffers['running_mean'] = new_rm.detach()
            self._buffers['running_var'] = new_rv.detach()
            
            # Normalize and apply affine
            eps_tensor = _scalar(self.eps, x.device).expand(reduced_shape)
            std_exp = (var + eps_tensor).sqrt().expand(shape)
            norm = diff / std_exp
            weight_exp = self.weight.tensor.expand(shape)
            bias_exp = self.bias.tensor.expand(shape)
            return (norm * weight_exp) + bias_exp
        else:
            # Evaluation mode uses running stats
            reduced_shape = [1, self.num_features, 1, 1]
            eps_tensor = _scalar(self.eps, x.device).expand(reduced_shape)
            rm_exp = self._buffers['running_mean'].expand(shape)
            rv_exp = self._buffers['running_var'] + eps_tensor
            std_exp = rv_exp.sqrt().expand(shape)
            norm = (x - rm_exp) / std_exp
            weight_exp = self.weight.tensor.expand(shape)
            bias_exp = self.bias.tensor.expand(shape)
            return (norm * weight_exp) + bias_exp
