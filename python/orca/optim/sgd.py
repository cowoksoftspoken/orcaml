"""Stochastic Gradient Descent optimizer with momentum and weight decay."""
from typing import Iterable

import orca
from orca.nn.parameter import Parameter

from .optimizer import Optimizer, _validate_non_negative_float


class SGD(Optimizer):
    """Implements stochastic gradient descent with optional momentum and weight decay.

    Nesterov momentum is **not** supported in this version.

    The update rule (with momentum and weight decay) is::

        v_t = momentum * v_{t-1} + grad + weight_decay * param
        param = param - lr * v_t

    Args:
        parameters: Iterable of parameters to optimize.
        lr: Learning rate. Default: ``0.01``.
        momentum: Momentum factor. Default: ``0.0`` (vanilla SGD).
        weight_decay: L2 penalty coefficient. Default: ``0.0`` (no penalty).
        dampening: Dampening for momentum. Default: ``0.0``.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
    ):
        super().__init__(parameters)
        self.lr = _validate_non_negative_float("lr", lr)
        self.momentum = _validate_non_negative_float("momentum", momentum)
        self.weight_decay = _validate_non_negative_float("weight_decay", weight_decay)
        self.dampening = _validate_non_negative_float("dampening", dampening)
        if self.dampening > 1.0:
            raise ValueError("dampening must be in the range [0, 1]")

        self._velocity = [None] * len(self.parameters)

    def step(self) -> None:
        """Performs a single optimization step."""
        with orca.no_grad():
            for parameter_index, parameter in enumerate(self.parameters):
                grad = parameter.tensor.grad()
                if grad is None:
                    continue

                if self.weight_decay != 0.0:
                    grad = grad + parameter.tensor * self.weight_decay

                if self.momentum != 0.0:
                    if self._velocity[parameter_index] is None:
                        self._velocity[parameter_index] = grad.detach()
                    else:
                        previous_velocity = self._velocity[parameter_index]
                        if self.dampening != 0.0:
                            self._velocity[parameter_index] = (
                                previous_velocity * self.momentum
                                + grad * (1.0 - self.dampening)
                            )
                        else:
                            self._velocity[parameter_index] = (
                                previous_velocity * self.momentum + grad
                            )

                    grad = self._velocity[parameter_index]

                update = grad * self.lr
                new_tensor = parameter.tensor - update

                new_leaf = new_tensor.detach()
                new_leaf.require_grad()
                parameter.update(new_leaf)
