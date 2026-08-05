from .nn.module import Module
from typing import Any

class DistributedDataParallel(Module):
    def __init__(
        self,
        module: Module,
        rank: int = 0,
        world_size: int = 1,
        master_addr: str = "127.0.0.1:18888",
        timeout: float = 30.0,
    ) -> None: ...
    def close(self) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def all_reduce_gradients(self) -> None: ...
