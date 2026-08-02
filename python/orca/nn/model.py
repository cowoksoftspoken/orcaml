from .module import Module

class Model(Module):
    """
    Base class for all neural network models in Orca.
    
    Subclassing ``nn.Model`` (which inherits from ``nn.Module``) provides
    a clean container to define layers and parameters. Orca prefers explicit
    and procedural training loops over implicit, framework-guided magic.
    
    Examples:
        >>> class MyModel(nn.Model):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc1 = nn.Linear(64, 32)
        ...         self.fc2 = nn.Linear(32, 10)
        ...     def forward(self, x):
        ...         return self.fc2(nn.ReLU()(self.fc1(x)))
    """
    def __init__(self) -> None:
        super().__init__()
