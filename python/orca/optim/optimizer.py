"""Base class for all optimizers."""
import math
from typing import Iterable
from orca.nn.parameter import Parameter


def _validate_finite_float(name: str, value: float) -> float:
    """Validate that a hyperparameter is a finite real number."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite float")

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite float") from exc

    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")

    return numeric


def _validate_non_negative_float(name: str, value: float) -> float:
    """Validate that a hyperparameter is finite and non-negative."""
    numeric = _validate_finite_float(name, value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _validate_positive_float(name: str, value: float) -> float:
    """Validate that a hyperparameter is finite and positive."""
    numeric = _validate_finite_float(name, value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _validate_probability(name: str, value: float) -> float:
    """Validate that a hyperparameter is in the half-open range [0, 1)."""
    numeric = _validate_finite_float(name, value)
    if numeric < 0.0 or numeric >= 1.0:
        raise ValueError(f"{name} must be in the range [0, 1)")
    return numeric


def _validate_betas(betas) -> tuple[float, float]:
    """Validate Adam-family beta coefficients."""
    try:
        beta1, beta2 = betas
    except (TypeError, ValueError) as exc:
        raise ValueError("betas must be a pair of floats") from exc

    return (
        _validate_probability("betas[0]", beta1),
        _validate_probability("betas[1]", beta2),
    )


def _validate_positive_int(name: str, value: int) -> int:
    """Validate that a hyperparameter is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class Optimizer:
    """Base class for all optimizers.

    Args:
        parameters: An iterable of ``Parameter`` objects to optimize.

    Raises:
        TypeError: If ``parameters`` is not an iterable of ``Parameter`` objects.
        ValueError: If ``parameters`` is empty.
    """

    def __init__(self, parameters: Iterable[Parameter]):
        try:
            self.parameters = list(parameters)
        except TypeError as exc:
            raise TypeError("parameters must be an iterable of Parameter objects") from exc

        if not self.parameters:
            raise ValueError("optimizer got an empty parameter list")

        for index, parameter in enumerate(self.parameters):
            if not isinstance(parameter, Parameter):
                raise TypeError(
                    f"parameters[{index}] must be an orca.nn.Parameter instance"
                )

    def zero_grad(self) -> None:
        """Clears the computational graph and all accumulated gradients.

        This resets the autograd tape so that the next forward pass
        builds a fresh computation graph. Only one call to the underlying
        tape clear is needed since all parameters share the same tape.
        """
        self.parameters[0].tensor.zero_grad()

    def step(self) -> None:
        """Performs a single optimization step (parameter update).

        Must be implemented by subclasses.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError
