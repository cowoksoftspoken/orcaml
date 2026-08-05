from os import fspath
from pathlib import Path

from .module import Module


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _scalar_value(tensor) -> float:
    values = tensor.to_list()
    if not values:
        raise ValueError("loss tensor did not contain any values")
    return float(values[0])


def _coerce_checkpoint_path(filepath) -> str:
    try:
        checkpoint_path = fspath(filepath)
    except TypeError as exc:
        raise TypeError("filepath must be a string or path-like object") from exc

    if not isinstance(checkpoint_path, str):
        raise TypeError("filepath must resolve to a string path")
    if not checkpoint_path:
        raise ValueError("filepath must not be empty")
    return checkpoint_path


def _normalize_verbose(verbose) -> int:
    if isinstance(verbose, bool):
        return 1 if verbose else 0
    if not isinstance(verbose, int):
        raise TypeError("verbose must be 0 or 1")
    if verbose not in (0, 1):
        raise ValueError("verbose must be 0 or 1")
    return verbose


def _rows_from_tensor(tensor):
    shape = list(tensor.shape)
    values = tensor.to_list()
    if not shape:
        raise ValueError("metric tensors must include a batch dimension")

    batch_size = shape[0]
    if batch_size == 0:
        return []

    row_width = 1
    for dimension in shape[1:]:
        row_width *= dimension

    return [
        values[row_start:row_start + row_width]
        for row_start in range(0, len(values), row_width)
    ]


def _argmax(values) -> int:
    if not values:
        raise ValueError("cannot compute argmax of an empty row")

    best_index = 0
    best_value = values[0]
    for value_index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = value_index
            best_value = value
    return best_index


def _accuracy(predictions, targets) -> float:
    prediction_rows = _rows_from_tensor(predictions)
    target_rows = _rows_from_tensor(targets)
    if len(prediction_rows) != len(target_rows):
        raise ValueError(
            "accuracy requires predictions and targets with the same batch size"
        )
    if not prediction_rows:
        return 0.0

    correct = 0
    for prediction_row, target_row in zip(prediction_rows, target_rows):
        predicted_class = _argmax(prediction_row)
        target_class = int(target_row[0]) if len(target_row) == 1 else _argmax(target_row)
        if predicted_class == target_class:
            correct += 1

    return correct / len(prediction_rows)


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
        self._optimizer = None
        self._loss_fn = None
        self._metrics = []
        self.stop_training = False

    def compile(self, optimizer="sgd", loss="mse", metrics=None, **optimizer_kwargs):
        """Configure a high-level training loop.

        Args:
            optimizer: Optimizer name (``"sgd"``, ``"adam"``, ``"adamw"``) or an
                optimizer instance.
            loss: Loss name (``"mse"``, ``"crossentropy"``) or loss module.
            metrics: Optional metric name or list of metric names. Currently
                supports ``"accuracy"``.
            **optimizer_kwargs: Hyperparameters passed to a named optimizer.

        Returns:
            Model: ``self`` for fluent setup.
        """
        self._optimizer = self._resolve_optimizer(optimizer, optimizer_kwargs)
        self._loss_fn = self._resolve_loss(loss)
        self._metrics = self._resolve_metrics(metrics)
        return self

    def fit(
        self,
        inputs,
        targets=None,
        *,
        epochs: int = 1,
        batch_size: int = 32,
        shuffle: bool = False,
        validation_data=None,
        callbacks=None,
        verbose: int = 0,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        """Train the model with a beginner-friendly high-level loop.

        Args:
            inputs: A ``DataLoader``, ``Dataset``, or input array.
            targets: Target array when ``inputs`` is not already batched.
            epochs: Number of training epochs.
            batch_size: Batch size used when arrays or datasets are passed.
            shuffle: Whether to shuffle arrays or datasets.
            validation_data: Optional ``DataLoader`` or ``(inputs, targets)`` pair.
            callbacks: Optional callback, callable, or sequence of callbacks.
            verbose: ``0`` for silent training, ``1`` for epoch logs.
            num_workers: Background sample-loading workers when Orca builds
                the ``DataLoader``.
            prefetch_factor: Prefetched batches per worker.

        Returns:
            dict: History containing epoch losses and configured metrics.
        """
        if self._optimizer is None or self._loss_fn is None:
            raise RuntimeError("Call model.compile(...) before model.fit(...)")

        epoch_count = _validate_positive_int("epochs", epochs)
        batch_size = _validate_positive_int("batch_size", batch_size)
        verbose = _normalize_verbose(verbose)
        loader = self._make_loader(
            inputs,
            targets,
            batch_size,
            shuffle,
            num_workers,
            prefetch_factor,
        )
        validation_loader = self._make_validation_loader(
            validation_data,
            batch_size,
            num_workers,
            prefetch_factor,
        )
        callback_list = self._make_callbacks(callbacks, verbose)
        callback_list.set_model(self)

        history = {"loss": []}
        for metric_name, _ in self._metrics:
            history[metric_name] = []
        if validation_loader is not None:
            history["val_loss"] = []
            for metric_name, _ in self._metrics:
                history[f"val_{metric_name}"] = []

        previous_training = self.training
        self.stop_training = False
        train_logs = {"epochs": epoch_count}
        callback_list.on_train_begin(train_logs)
        self.train(True)
        training_error = None
        try:
            for epoch_index in range(epoch_count):
                if self.stop_training:
                    break
                callback_list.on_epoch_begin(epoch_index, train_logs)
                if self.stop_training:
                    break

                epoch_result = self._run_epoch(loader)
                epoch_logs = self._flatten_evaluation_result(epoch_result)
                for history_key, history_value in epoch_logs.items():
                    history[history_key].append(history_value)

                if validation_loader is not None:
                    validation_result = self._evaluate_in_mode(validation_loader)
                    validation_logs = self._flatten_evaluation_result(validation_result)
                    for metric_name, metric_value in validation_logs.items():
                        history_key = f"val_{metric_name}"
                        epoch_logs[history_key] = metric_value
                        history[history_key].append(metric_value)

                callback_list.on_epoch_end(epoch_index, epoch_logs)
        except BaseException as exc:
            training_error = exc
            raise
        finally:
            self.train(previous_training)
            end_logs = {"epochs": epoch_count, "history": history}
            if training_error is None:
                callback_list.on_train_end(end_logs)
            else:
                try:
                    callback_list.on_train_end(end_logs)
                except Exception:
                    pass

        return history

    def evaluate(
        self,
        inputs,
        targets=None,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        """Evaluate loss and metrics without mutating the current train/eval mode.

        Args:
            inputs: A supervised ``DataLoader``, ``Dataset``, or input array.
            targets: Target array when ``inputs`` is not already supervised.
            batch_size: Batch size used when arrays or datasets are passed.
            num_workers: Background sample-loading workers when Orca builds
                the ``DataLoader``.
            prefetch_factor: Prefetched batches per worker.

        Returns:
            dict: ``{"loss": value, ...metrics}``.
        """
        if self._loss_fn is None:
            raise RuntimeError("Call model.compile(...) before model.evaluate(...)")

        batch_size = _validate_positive_int("batch_size", batch_size)
        loader = self._make_loader(
            inputs,
            targets,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        result = self._evaluate_in_mode(loader)
        return self._flatten_evaluation_result(result)

    def predict(
        self,
        inputs,
        *,
        batch_size: int = 32,
        input_count=None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        """Run inference with ``no_grad`` and restore the current train/eval mode.

        Args:
            inputs: A ``DataLoader``, ``Dataset``, or input array.
            batch_size: Batch size used when arrays or datasets are passed.
            input_count: Optional number of tuple fields to pass into ``forward``.
                By default, one-field batches are passed through, two-field
                batches are treated as ``(inputs, targets)``, and larger tuples
                are treated as multi-input batches.
            num_workers: Background sample-loading workers when Orca builds
                the ``DataLoader``.
            prefetch_factor: Prefetched batches per worker.

        Returns:
            Tensor | list[Tensor]: A single tensor for one emitted batch, or a
            list of tensors when multiple batches are produced.
        """
        batch_size = _validate_positive_int("batch_size", batch_size)
        normalized_input_count = self._normalize_input_count(input_count)
        loader = self._make_prediction_loader(
            inputs,
            batch_size,
            num_workers,
            prefetch_factor,
        )
        predictions = self._predict_in_mode(loader, normalized_input_count)
        if len(predictions) == 1:
            return predictions[0]
        return predictions

    def save(self, filepath):
        """Save model weights to a Safetensors checkpoint.

        Args:
            filepath: String or path-like checkpoint path.

        Returns:
            Model: ``self`` for fluent workflows.
        """
        checkpoint_path = _coerce_checkpoint_path(filepath)
        parent = Path(checkpoint_path).parent
        if str(parent) and not parent.exists():
            raise FileNotFoundError(f"checkpoint directory does not exist: {parent}")
        self.save_weights(checkpoint_path)
        return self

    def load(self, filepath):
        """Load model weights from a Safetensors checkpoint.

        Args:
            filepath: String or path-like checkpoint path.

        Returns:
            Model: ``self`` for fluent workflows.
        """
        checkpoint_path = _coerce_checkpoint_path(filepath)
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")
        self.load_weights(checkpoint_path)
        return self

    def _resolve_optimizer(self, optimizer, optimizer_kwargs):
        import orca.optim as optim

        if isinstance(optimizer, str):
            optimizer_name = optimizer.lower()
            optimizer_classes = {
                "sgd": optim.SGD,
                "adam": optim.Adam,
                "adamw": optim.AdamW,
            }
            if optimizer_name not in optimizer_classes:
                raise ValueError(
                    "Unknown optimizer "
                    f"{optimizer!r}; expected one of {sorted(optimizer_classes)}"
                )
            return optimizer_classes[optimizer_name](
                self.parameters(),
                **optimizer_kwargs,
            )

        if optimizer_kwargs:
            raise ValueError("optimizer_kwargs are only valid with named optimizers")
        if not hasattr(optimizer, "zero_grad") or not hasattr(optimizer, "step"):
            raise TypeError("optimizer must be a supported name or optimizer instance")
        return optimizer

    @staticmethod
    def _resolve_loss(loss):
        import orca.nn as nn

        if isinstance(loss, str):
            loss_name = loss.lower()
            loss_classes = {
                "mse": nn.MSELoss,
                "mse_loss": nn.MSELoss,
                "crossentropy": nn.CrossEntropyLoss,
                "cross_entropy": nn.CrossEntropyLoss,
                "cross_entropy_loss": nn.CrossEntropyLoss,
            }
            if loss_name not in loss_classes:
                raise ValueError(
                    f"Unknown loss {loss!r}; expected one of {sorted(loss_classes)}"
                )
            return loss_classes[loss_name]()

        if not callable(loss):
            raise TypeError("loss must be a supported name or callable loss module")
        return loss

    @staticmethod
    def _resolve_metrics(metrics):
        if metrics is None:
            return []
        if isinstance(metrics, str):
            metrics = [metrics]
        elif callable(metrics):
            metrics = [metrics]

        resolved_metrics = []
        for metric in metrics:
            if callable(metric):
                metric_name = getattr(metric, "__name__", "metric")
                resolved_metrics.append((metric_name, metric))
            elif isinstance(metric, str):
                metric_name = metric.lower()
                if metric_name != "accuracy":
                    raise ValueError("Only the 'accuracy' metric is currently supported")
                resolved_metrics.append(("accuracy", _accuracy))
            else:
                raise TypeError("metrics must be a string, callable, sequence, or None")
        return resolved_metrics

    @staticmethod
    def _make_loader(
        inputs,
        targets,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        prefetch_factor: int,
    ):
        import orca.data as data

        if isinstance(inputs, data.DataLoader):
            if targets is not None:
                raise ValueError("targets must be None when inputs is a DataLoader")
            if num_workers != 0 or prefetch_factor != 2:
                raise ValueError(
                    "num_workers and prefetch_factor are only valid when "
                    "model creates the DataLoader"
                )
            return inputs

        if isinstance(inputs, data.Dataset):
            if targets is not None:
                raise ValueError("targets must be None when inputs is a Dataset")
            return data.DataLoader(
                inputs,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

        if targets is None:
            raise ValueError("targets are required when inputs is not a DataLoader")

        return data.from_arrays(
            inputs,
            targets,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    @staticmethod
    def _make_validation_loader(
        validation_data,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
    ):
        import orca.data as data

        if validation_data is None:
            return None
        if isinstance(validation_data, data.DataLoader):
            if num_workers != 0 or prefetch_factor != 2:
                raise ValueError(
                    "num_workers and prefetch_factor are only valid when "
                    "model creates the validation DataLoader"
                )
            return validation_data
        if (
            isinstance(validation_data, tuple)
            and len(validation_data) == 2
        ):
            return data.from_arrays(
                validation_data[0],
                validation_data[1],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
        raise TypeError("validation_data must be a DataLoader or (inputs, targets) tuple")

    @staticmethod
    def _make_prediction_loader(
        inputs,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
    ):
        import orca.data as data

        if isinstance(inputs, data.DataLoader):
            if num_workers != 0 or prefetch_factor != 2:
                raise ValueError(
                    "num_workers and prefetch_factor are only valid when "
                    "model creates the prediction DataLoader"
                )
            return inputs
        if isinstance(inputs, data.Dataset):
            return data.DataLoader(
                inputs,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
        return data.from_arrays(
            inputs,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    @staticmethod
    def _make_callbacks(callbacks, verbose: int):
        from orca.callbacks import _normalize_callbacks

        return _normalize_callbacks(callbacks, verbose=verbose)

    def _run_epoch(self, loader):
        total_loss = 0.0
        total_batches = 0
        metric_totals = {metric_name: 0.0 for metric_name, _ in self._metrics}

        for batch in loader:
            batch_inputs, batch_targets = self._split_batch(batch)
            self._optimizer.zero_grad()
            predictions = self(*batch_inputs)
            loss = self._loss_fn(predictions, batch_targets)
            loss.backward()
            self._optimizer.step()

            total_loss += _scalar_value(loss)
            total_batches += 1
            for metric_name, metric_fn in self._metrics:
                metric_totals[metric_name] += metric_fn(predictions, batch_targets)

        if total_batches == 0:
            raise ValueError("training data produced no batches")

        return {
            "loss": total_loss / total_batches,
            "metrics": {
                metric_name: metric_total / total_batches
                for metric_name, metric_total in metric_totals.items()
            },
        }

    def _evaluate(self, loader):
        import orca

        total_loss = 0.0
        total_batches = 0
        metric_totals = {metric_name: 0.0 for metric_name, _ in self._metrics}

        with orca.no_grad():
            for batch in loader:
                batch_inputs, batch_targets = self._split_batch(batch)
                predictions = self(*batch_inputs)
                loss = self._loss_fn(predictions, batch_targets)

                total_loss += _scalar_value(loss)
                total_batches += 1
                for metric_name, metric_fn in self._metrics:
                    metric_totals[metric_name] += metric_fn(predictions, batch_targets)

        if total_batches == 0:
            raise ValueError("validation data produced no batches")

        return {
            "loss": total_loss / total_batches,
            "metrics": {
                metric_name: metric_total / total_batches
                for metric_name, metric_total in metric_totals.items()
            },
        }

    def _evaluate_in_mode(self, loader):
        previous_training = self.training
        self.eval()
        try:
            return self._evaluate(loader)
        finally:
            self.train(previous_training)

    def _predict_in_mode(self, loader, input_count):
        import orca

        predictions = []
        previous_training = self.training
        self.eval()
        try:
            with orca.no_grad():
                for batch in loader:
                    batch_inputs = self._split_prediction_batch(batch, input_count)
                    predictions.append(self(*batch_inputs))
        finally:
            self.train(previous_training)

        if not predictions:
            raise ValueError("prediction data produced no batches")
        return predictions

    @staticmethod
    def _flatten_evaluation_result(result):
        flattened = {"loss": result["loss"]}
        flattened.update(result["metrics"])
        return flattened

    @staticmethod
    def _split_batch(batch):
        if not isinstance(batch, tuple) or len(batch) < 2:
            raise ValueError("model.fit expects batches shaped as (inputs, targets)")

        batch_inputs = batch[:-1]
        batch_targets = batch[-1]
        return batch_inputs, batch_targets

    @staticmethod
    def _normalize_input_count(input_count):
        if input_count is None:
            return None
        return _validate_positive_int("input_count", input_count)

    @staticmethod
    def _split_prediction_batch(batch, input_count):
        if not isinstance(batch, tuple):
            if input_count is not None and input_count != 1:
                raise ValueError("input_count exceeds fields available in prediction batch")
            return (batch,)
        if not batch:
            raise ValueError("prediction batch did not contain inputs")

        if input_count is not None:
            if input_count > len(batch):
                raise ValueError("input_count exceeds fields available in prediction batch")
            return batch[:input_count]
        if len(batch) <= 2:
            return (batch[0],)
        return batch
