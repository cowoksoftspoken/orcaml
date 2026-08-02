import math
from typing import Optional
import orca
from orca.tensor import Tensor, DType
from .module import Module
from .linear import Linear

class QuantizedLinear(Module):
    """
    Quantized version of the Linear layer using 8-bit symmetric quantization.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_enabled = bias
        
        self.weight = orca.Tensor.zeros([in_features, out_features])
        self.bias = orca.Tensor.zeros([1, out_features]) if bias else None
        
        self.scale_w = 1.0
        self.weight_q = None
        self.quantized = False

    @classmethod
    def from_float(cls, float_linear: Linear):
        """
        Creates a QuantizedLinear layer from a trained float Linear layer.
        """
        q_linear = cls(float_linear.in_features, float_linear.out_features, float_linear.bias is not None)
        q_linear.weight = float_linear.weight.tensor
        if float_linear.bias is not None:
            q_linear.bias = float_linear.bias.tensor
        q_linear.quantize()
        return q_linear

    def quantize(self):
        """
        Calibrates and quantizes the weight tensor to INT8 symmetrically.
        """
        w_list = self.weight.to_list()
        max_val = max(abs(x) for x in w_list) if w_list else 1e-5
        if max_val == 0:
            max_val = 1e-5
            
        self.scale_w = max_val / 127.0
        
        self.weight_q = []
        for x in w_list:
            q_val = round(x / self.scale_w)
            q_val = max(-128, min(127, q_val))
            self.weight_q.append(q_val)
            
        self.weight_q_tensor = Tensor.from_list(
            [float(q) for q in self.weight_q],
            shape=self.weight.shape,
            device=self.weight.device
        )
        self.quantized = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.quantized:
            out = x @ self.weight
            if self.bias is not None:
                out = out + self.bias.expand(out.shape)
            return out
            
        # 1. Quantize input x dynamically
        x_list = x.to_list()
        max_x = max(abs(val) for val in x_list) if x_list else 1e-5
        if max_x == 0:
            max_x = 1e-5
        scale_x = max_x / 127.0
        
        x_q = [max(-128.0, min(127.0, float(round(val / scale_x)))) for val in x_list]
        x_q_tensor = Tensor.from_list(x_q, shape=x.shape, device=x.device)
        
        # Ensure weight_q_tensor is on the correct device
        if str(self.weight_q_tensor.device) != str(x.device):
            self.weight_q_tensor = self.weight_q_tensor.to(str(x.device))
            
        # 2. Perform matrix multiplication in Rust
        out_q_tensor = x_q_tensor @ self.weight_q_tensor
        
        # 3. Dequantize output in Rust
        scale_out = scale_x * self.scale_w
        out_tensor = out_q_tensor * scale_out
        
        if self.bias is not None:
            out_tensor = out_tensor + self.bias.expand(out_tensor.shape)
            
        return out_tensor
