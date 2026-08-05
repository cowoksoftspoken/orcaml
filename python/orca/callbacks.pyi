from typing import Any, Callable, Optional, Sequence


class Callback:
    model: Any
    def set_model(self, model: Any) -> None: ...
    def on_train_begin(self, logs: Optional[dict] = None) -> None: ...
    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None: ...
    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None: ...
    def on_train_end(self, logs: Optional[dict] = None) -> None: ...


class LambdaCallback(Callback):
    def __init__(
        self,
        *,
        on_train_begin: Optional[Callable[[dict], None]] = None,
        on_epoch_begin: Optional[Callable[[int, dict], None]] = None,
        on_epoch_end: Optional[Callable[[int, dict], None]] = None,
        on_train_end: Optional[Callable[[dict], None]] = None,
    ) -> None: ...


class ConsoleLogger(Callback):
    def __init__(
        self,
        *,
        stream: Any = None,
        every: int = 1,
        precision: int = 4,
    ) -> None: ...


class CSVLogger(Callback):
    def __init__(self, filepath: Any, *, append: bool = False) -> None: ...
    def close(self) -> None: ...


__all__: Sequence[str]
