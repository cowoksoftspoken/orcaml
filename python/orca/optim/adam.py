import orca
from typing import Iterable, Tuple
from .optimizer import (
    Optimizer,
    _validate_betas,
    _validate_non_negative_float,
    _validate_positive_float,
)
from orca.nn.parameter import Parameter


class Adam(Optimizer):
    """
    Implements Adam algorithm.
    
    Args:
        parameters (Iterable[Parameter]): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. Default: 0.001.
        betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square. Default: (0.9, 0.999).
        eps (float, optional): term added to the denominator to improve numerical stability. Default: 1e-8.
    """
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(parameters)
        self.lr = _validate_non_negative_float("lr", lr)
        self.beta1, self.beta2 = _validate_betas(betas)
        self.eps = _validate_positive_float("eps", eps)
        self.t = 0
        
        # Initialize state
        self.m = []
        self.v = []
        for parameter in self.parameters:
            shape = parameter.tensor.shape
            device = parameter.tensor.device
            dtype = parameter.tensor.dtype
            self.m.append(orca.Tensor.zeros(shape, dtype=dtype, device=device))
            self.v.append(orca.Tensor.zeros(shape, dtype=dtype, device=device))

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        grads = []
        for parameter_index, parameter in enumerate(self.parameters):
            if not parameter.tensor.requires_grad:
                continue

            grad = parameter.tensor.grad()
            if grad is not None:
                grads.append((parameter_index, parameter, grad))

        if not grads:
            return

        self.t += 1

        with orca.no_grad():
            for parameter_index, parameter, grad in grads:
                device = parameter.tensor.device
                dtype = parameter.tensor.dtype
                
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                m_prev = self.m[parameter_index]
                m_new = m_prev * self.beta1 + grad * (1.0 - self.beta1)
                
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v_prev = self.v[parameter_index]
                v_new = v_prev * self.beta2 + (grad * grad) * (1.0 - self.beta2)
                
                self.m[parameter_index] = m_new
                self.v[parameter_index] = v_new
                
                # Bias correction
                m_hat = m_new * (1.0 / (1.0 - self.beta1**self.t))
                v_hat = v_new * (1.0 / (1.0 - self.beta2**self.t))
                
                # Parameter update: p = p - lr * m_hat / (sqrt(v_hat) + eps)
                eps_tensor = orca.Tensor.scalar(self.eps, dtype=dtype, device=device).expand(v_hat.shape)
                denom = v_hat.sqrt() + eps_tensor
                
                update = (m_hat / denom) * self.lr
                new_tensor = parameter.tensor - update
                
                # Preserve requires_grad manually by detaching and re-enabling graph tracking
                new_leaf = new_tensor.detach()
                new_leaf.require_grad()
                parameter.update(new_leaf)
