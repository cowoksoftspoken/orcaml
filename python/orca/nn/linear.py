import math
from typing import Optional
from orca.tensor import Tensor
from .module import Module
from .parameter import Parameter

class Linear(Module):
    """
    Applies a linear transformation to the incoming data: `y = x A^T + b`.
    
    This module supports batched input. The input tensor is flattened up to the last dimension, 
    the transformation is applied, and then reshaped back.
    
    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True.
        
    Attributes:
        weight (Parameter): the learnable weights of the module of shape `(in_features, out_features)`.
        bias (Optional[Parameter]): the learnable bias of the module of shape `(1, out_features)`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # PyTorch-compatible Kaiming Uniform default weight initialization
        self.weight = Parameter(Tensor.zeros([in_features, out_features], requires_grad=True))
        from . import init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5.0), mode='fan_in', nonlinearity='leaky_relu')
        
        if bias:
            self.bias = Parameter(Tensor.zeros([1, out_features], requires_grad=True))
            bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Linear layer.
        
        Args:
            x (Tensor): Input tensor of shape `(..., in_features)`.
            
        Returns:
            Tensor: Output tensor of shape `(..., out_features)`.
        """
        original_shape = x.shape
        rank = len(original_shape)
        
        if rank > 2:
            # Flatten to 2D: [batch * seq_len * ..., in_features]
            flat_size = 1
            for d in original_shape[:-1]:
                flat_size *= d
            flat_shape = [flat_size, self.in_features]
            x_flat = x.reshape(flat_shape)
        else:
            x_flat = x
            
        from orca import autocast
        if autocast.is_enabled():
            target_dtype = autocast.current_dtype()
            if x_flat.dtype != target_dtype:
                x_flat = x_flat.to_dtype(target_dtype)
            weight = self.weight.tensor.to_dtype(target_dtype)
            bias = self.bias.tensor.to_dtype(target_dtype) if self.bias is not None else None
        else:
            weight = self.weight.tensor
            bias = self.bias.tensor if self.bias is not None else None

        out = x_flat @ weight
        
        if bias is not None:
            # expand bias to match out shape and add
            b = bias.expand(out.shape)
            out = out + b
            
        if rank > 2:
            # Reshape back to [batch, seq_len, ..., out_features]
            new_shape = list(original_shape[:-1]) + [self.out_features]
            out = out.reshape(new_shape)
            
        return out
