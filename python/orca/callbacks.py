import csv
import sys
from os import fspath
from pathlib import Path


_HOOK_NAMES = (
    "on_train_begin",
    "on_epoch_begin",
    "on_epoch_end",
    "on_train_end",
)


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _coerce_path(filepath) -> str:
    try:
        resolved_path = fspath(filepath)
    except TypeError as exc:
        raise TypeError("filepath must be a string or path-like object") from exc

    if not isinstance(resolved_path, str):
        raise TypeError("filepath must resolve to a string path")
    if not resolved_path:
        raise ValueError("filepath must not be empty")
    return resolved_path


def _coerce_logs(logs):
    if logs is None:
        return {}
    return dict(logs)


def _format_metric(value, precision: int) -> str:
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _is_callback_like(callback) -> bool:
    return any(callable(getattr(callback, hook_name, None)) for hook_name in _HOOK_NAMES)


class Callback:
    """Base class for high-level training callbacks.

    Subclass this when you need custom telemetry, checkpointing, or training
    control around ``Model.fit``.
    """

    def set_model(self, model) -> None:
        self.model = model

    def on_train_begin(self, logs=None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        pass

    def on_train_end(self, logs=None) -> None:
        pass


class LambdaCallback(Callback):
    """Create a callback from hook callables."""

    def __init__(
        self,
        *,
        on_train_begin=None,
        on_epoch_begin=None,
        on_epoch_end=None,
        on_train_end=None,
    ):
        hooks = {
            "on_train_begin": on_train_begin,
            "on_epoch_begin": on_epoch_begin,
            "on_epoch_end": on_epoch_end,
            "on_train_end": on_train_end,
        }
        if not any(hook is not None for hook in hooks.values()):
            raise ValueError("LambdaCallback requires at least one hook")
        for hook_name, hook in hooks.items():
            if hook is not None and not callable(hook):
                raise TypeError(f"{hook_name} must be callable or None")

        self._on_train_begin = on_train_begin
        self._on_epoch_begin = on_epoch_begin
        self._on_epoch_end = on_epoch_end
        self._on_train_end = on_train_end

    def on_train_begin(self, logs=None) -> None:
        if self._on_train_begin is not None:
            self._on_train_begin(_coerce_logs(logs))

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        if self._on_epoch_begin is not None:
            self._on_epoch_begin(epoch, _coerce_logs(logs))

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if self._on_epoch_end is not None:
            self._on_epoch_end(epoch, _coerce_logs(logs))

    def on_train_end(self, logs=None) -> None:
        if self._on_train_end is not None:
            self._on_train_end(_coerce_logs(logs))


class ConsoleLogger(Callback):
    """Print one training summary line per epoch."""

    def __init__(self, *, stream=None, every: int = 1, precision: int = 4):
        self.stream = sys.stdout if stream is None else stream
        self.every = _validate_positive_int("every", every)
        self.precision = _validate_non_negative_int("precision", precision)
        self._epochs = None

    def on_train_begin(self, logs=None) -> None:
        self._epochs = _coerce_logs(logs).get("epochs")

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        epoch_number = epoch + 1
        if epoch_number % self.every != 0 and epoch_number != self._epochs:
            return

        metric_parts = [
            f"{metric_name}={_format_metric(metric_value, self.precision)}"
            for metric_name, metric_value in _coerce_logs(logs).items()
        ]
        metrics = " - ".join(metric_parts)
        suffix = f" - {metrics}" if metrics else ""
        print(f"Epoch {epoch_number}/{self._epochs}{suffix}", file=self.stream)
        self.stream.flush()


class CSVLogger(Callback):
    """Write epoch metrics to a CSV file."""

    def __init__(self, filepath, *, append: bool = False):
        self.filepath = _coerce_path(filepath)
        self.append = bool(append)
        self._file = None
        self._writer = None
        self._fieldnames = None

    def on_train_begin(self, logs=None) -> None:
        parent = Path(self.filepath).parent
        if str(parent) and not parent.exists():
            raise FileNotFoundError(f"metrics directory does not exist: {parent}")

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        row = {"epoch": epoch + 1}
        row.update(_coerce_logs(logs))
        if self._writer is None:
            self._open_writer(list(row.keys()))
        self._writer.writerow(row)
        self._file.flush()

    def on_train_end(self, logs=None) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def _open_writer(self, fieldnames) -> None:
        path = Path(self.filepath)
        has_existing_content = path.exists() and path.stat().st_size > 0
        if self.append and has_existing_content:
            existing_fieldnames = self._read_existing_fieldnames(path)
            if existing_fieldnames != fieldnames:
                raise ValueError(
                    "CSVLogger cannot append metrics with a different header"
                )
            self._file = path.open("a", newline="", encoding="utf-8")
            self._fieldnames = existing_fieldnames
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            return

        self._file = path.open("w", newline="", encoding="utf-8")
        self._fieldnames = fieldnames
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()

    @staticmethod
    def _read_existing_fieldnames(path: Path):
        with path.open("r", newline="", encoding="utf-8") as existing_file:
            reader = csv.reader(existing_file)
            try:
                return next(reader)
            except StopIteration:
                return []


class _CallbackList:
    def __init__(self, callbacks):
        self.callbacks = list(callbacks)

    def set_model(self, model) -> None:
        for callback in self.callbacks:
            set_model = getattr(callback, "set_model", None)
            if callable(set_model):
                set_model(model)
            else:
                callback.model = model

    def on_train_begin(self, logs=None) -> None:
        self._call("on_train_begin", _coerce_logs(logs))

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self._call("on_epoch_begin", epoch, _coerce_logs(logs))

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        self._call("on_epoch_end", epoch, _coerce_logs(logs))

    def on_train_end(self, logs=None) -> None:
        self._call("on_train_end", _coerce_logs(logs))

    def _call(self, hook_name: str, *args) -> None:
        for callback in self.callbacks:
            hook = getattr(callback, hook_name, None)
            if callable(hook):
                hook(*args)


def _normalize_callbacks(callbacks, *, verbose: int = 0) -> _CallbackList:
    normalized_callbacks = []
    if callbacks is None:
        pass
    elif callable(callbacks) and not _is_callback_like(callbacks):
        normalized_callbacks.append(LambdaCallback(on_epoch_end=callbacks))
    elif _is_callback_like(callbacks):
        normalized_callbacks.append(callbacks)
    else:
        try:
            callback_items = list(callbacks)
        except TypeError as exc:
            raise TypeError("callbacks must be a callback, callable, sequence, or None") from exc

        for callback in callback_items:
            if callable(callback) and not _is_callback_like(callback):
                normalized_callbacks.append(LambdaCallback(on_epoch_end=callback))
            elif _is_callback_like(callback):
                normalized_callbacks.append(callback)
            else:
                raise TypeError(
                    "callbacks must contain callback-like objects or callables"
                )

    if verbose:
        normalized_callbacks.append(ConsoleLogger())
    return _CallbackList(normalized_callbacks)


__all__ = ["Callback", "LambdaCallback", "ConsoleLogger", "CSVLogger", "_normalize_callbacks"]
