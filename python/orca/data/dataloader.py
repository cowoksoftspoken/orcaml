"""Dataset and DataLoader utilities for Orca's Python API."""
import csv
import random
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Generic, Iterator, List, Optional, TypeVar, Union

import orca


T_co = TypeVar("T_co", covariant=True)
ColumnSelector = Union[int, str]
ColumnSelectors = Optional[Union[ColumnSelector, Sequence[ColumnSelector]]]


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _to_plain_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if _is_sequence(value):
        return [_to_plain_list(item) for item in value]
    return float(value)


def _shape_of(value: Any) -> List[int]:
    if not _is_sequence(value):
        return []
    if len(value) == 0:
        return [0]

    first_shape = _shape_of(value[0])
    for item_index, item in enumerate(value[1:], start=1):
        item_shape = _shape_of(item)
        if item_shape != first_shape:
            raise ValueError(
                "Samples must be rectangular; "
                f"item at index {item_index} has shape {item_shape}, expected {first_shape}"
            )
    return [len(value)] + first_shape


def _flatten(value: Any) -> List[float]:
    if not _is_sequence(value):
        return [float(value)]

    flattened = []
    for item in value:
        flattened.extend(_flatten(item))
    return flattened


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _reshape(flat_values: Sequence[float], shape: Sequence[int]) -> Any:
    if not shape:
        if len(flat_values) != 1:
            raise ValueError("Scalar reshape requires exactly one value")
        return float(flat_values[0])

    if len(shape) == 1:
        return [float(value) for value in flat_values]

    stride = _product(shape[1:])
    return [
        _reshape(flat_values[start:start + stride], shape[1:])
        for start in range(0, len(flat_values), stride)
    ]


def _tensor_to_samples(tensor) -> List[Any]:
    shape = list(tensor.shape)
    if not shape:
        raise ValueError("ArrayDataset requires tensors with at least one dimension")

    flat_values = tensor.to_list()
    if len(shape) == 1:
        return [[float(value)] for value in flat_values]

    sample_shape = shape[1:]
    sample_size = _product(sample_shape)
    return [
        _reshape(flat_values[start:start + sample_size], sample_shape)
        for start in range(0, len(flat_values), sample_size)
    ]


def _array_to_samples(array: Any) -> List[Any]:
    if hasattr(array, "to_list") and hasattr(array, "shape"):
        return _tensor_to_samples(array)

    values = _to_plain_list(array)
    if not _is_sequence(values):
        raise ValueError("ArrayDataset arrays must be sequences")
    if len(values) == 0:
        raise ValueError("ArrayDataset arrays cannot be empty")
    if not _is_sequence(values[0]):
        return [[float(value)] for value in values]
    return values


def _as_column_list(columns: ColumnSelectors, name: str) -> Optional[List[ColumnSelector]]:
    if columns is None:
        return None
    if isinstance(columns, bool):
        raise TypeError(f"{name} must be an int, str, or sequence of int/str")
    if isinstance(columns, (int, str)):
        return [columns]
    if not _is_sequence(columns):
        raise TypeError(f"{name} must be an int, str, or sequence of int/str")

    result = list(columns)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    for column in result:
        if not isinstance(column, (int, str)) or isinstance(column, bool):
            raise TypeError(f"{name} must contain only int or str selectors")
    return result


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


def _normalize_device(device):
    if isinstance(device, str):
        return orca.Device(device)
    return device


class Dataset(Generic[T_co]):
    """Abstract map-style dataset."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> T_co:
        raise NotImplementedError


class DataLoader:
    """Batch a map-style dataset into Orca tensors.

    Args:
        dataset: Dataset from which to load samples.
        batch_size: Number of samples per batch.
        shuffle: Whether to reshuffle sample order for each iteration.
        drop_last: Whether to drop the final incomplete batch.
        seed: Optional deterministic shuffle seed.
        dtype: Optional dtype applied to emitted tensors.
        device: Optional device applied to emitted tensors.
        num_workers: Number of background threads used to load dataset samples.
            Defaults to 0 for fully synchronous loading.
        prefetch_factor: Number of prefetched batches per worker when
            ``num_workers`` is greater than 0.

    Worker threads load dataset samples only. Collation and tensor creation stay
    on the caller thread so backend operations remain thread-local and yielded
    batch order is deterministic.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
        dtype=None,
        device=None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError("dataset must implement __len__ and __getitem__")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool")
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a bool")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer or None")

        self.dataset = dataset
        self.batch_size = _validate_positive_int("batch_size", batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.dtype = dtype
        self.device = _normalize_device(device)
        self.num_workers = _validate_non_negative_int("num_workers", num_workers)
        self.prefetch_factor = _validate_positive_int("prefetch_factor", prefetch_factor)

        self._dataset_length = len(dataset)
        if self._dataset_length < 0:
            raise ValueError("dataset length cannot be negative")

    def __len__(self) -> int:
        if self.drop_last:
            return self._dataset_length // self.batch_size
        return (self._dataset_length + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        batch_indices_list = self._make_batch_indices()
        if self.num_workers == 0:
            yield from self._iter_sequential(batch_indices_list)
            return

        yield from self._iter_prefetched(batch_indices_list)

    def _make_batch_indices(self) -> List[List[int]]:
        indices = list(range(self._dataset_length))
        if self.shuffle:
            random_source = random.Random(self.seed)
            random_source.shuffle(indices)

        batch_indices_list = []
        for batch_start in range(0, len(indices), self.batch_size):
            batch_indices = indices[batch_start:batch_start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            batch_indices_list.append(batch_indices)
        return batch_indices_list

    def _iter_sequential(self, batch_indices_list: List[List[int]]) -> Iterator[Any]:
        for batch_indices in batch_indices_list:
            yield self._collate(self._load_samples(batch_indices))

    def _iter_prefetched(self, batch_indices_list: List[List[int]]) -> Iterator[Any]:
        if not batch_indices_list:
            return

        prefetch_limit = max(1, self.num_workers * self.prefetch_factor)
        next_submit_index = 0
        next_yield_index = 0
        completed_batches = {}
        futures_by_batch_index = {}

        with ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="orca-dataloader",
        ) as executor:

            def submit_until_prefetched() -> None:
                nonlocal next_submit_index
                while (
                    next_submit_index < len(batch_indices_list)
                    and len(futures_by_batch_index) < prefetch_limit
                ):
                    future = executor.submit(
                        self._load_samples,
                        batch_indices_list[next_submit_index],
                    )
                    futures_by_batch_index[future] = next_submit_index
                    next_submit_index += 1

            try:
                submit_until_prefetched()
                while next_yield_index < len(batch_indices_list):
                    if next_yield_index not in completed_batches:
                        done_futures, _ = wait(
                            futures_by_batch_index,
                            return_when=FIRST_COMPLETED,
                        )
                        for future in done_futures:
                            batch_index = futures_by_batch_index.pop(future)
                            completed_batches[batch_index] = future.result()
                        submit_until_prefetched()

                    while next_yield_index in completed_batches:
                        yield self._collate(completed_batches.pop(next_yield_index))
                        next_yield_index += 1
                        submit_until_prefetched()
            finally:
                for future in futures_by_batch_index:
                    future.cancel()

    def _load_samples(self, batch_indices: List[int]) -> List[Any]:
        samples = []
        for sample_index in batch_indices:
            try:
                samples.append(self.dataset[sample_index])
            except Exception as exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        f"DataLoader failed while loading dataset index {sample_index}"
                    )
                raise
        return samples

    def _collate(self, samples: List[Any]) -> Any:
        normalized_samples = [
            sample if isinstance(sample, tuple) else (sample,)
            for sample in samples
        ]

        arity = len(normalized_samples[0])
        for sample_index, sample in enumerate(normalized_samples[1:], start=1):
            if len(sample) != arity:
                raise ValueError(
                    f"Sample at index {sample_index} has {len(sample)} fields, expected {arity}"
                )

        tensors = []
        for field_index in range(arity):
            field_values = [
                _to_plain_list(sample[field_index])
                for sample in normalized_samples
            ]
            sample_shape = _shape_of(field_values[0])
            for sample_index, value in enumerate(field_values[1:], start=1):
                value_shape = _shape_of(value)
                if value_shape != sample_shape:
                    raise ValueError(
                        "All samples in a batch must have the same shape; "
                        f"field {field_index} sample {sample_index} has shape {value_shape}, "
                        f"expected {sample_shape}"
                    )

            tensor_shape = [len(samples)] + sample_shape
            tensor = orca.Tensor.from_list(
                _flatten(field_values),
                shape=tensor_shape,
                device=self.device,
            )
            if self.dtype is not None:
                tensor = tensor.to_dtype(self.dtype)
            tensors.append(tensor)

        if isinstance(samples[0], tuple):
            return tuple(tensors)
        return tensors[0]


class ArrayDataset(Dataset[tuple[Any, ...]]):
    """Dataset wrapper for in-memory arrays.

    Args:
        *arrays: One or more equally-sized arrays, numpy arrays, or Orca tensors.
        one_hot_classes: Optional number of classes for one-hot encoding the
            final array.
    """

    def __init__(self, *arrays: Any, one_hot_classes: Optional[int] = None):
        if not arrays:
            raise ValueError("At least one array must be provided to ArrayDataset")

        self.arrays = [_array_to_samples(array) for array in arrays]
        self.length = len(self.arrays[0])
        for array_index, array in enumerate(self.arrays):
            if len(array) != self.length:
                raise ValueError(
                    f"Array at index {array_index} has length {len(array)}, "
                    f"expected {self.length}"
                )

        if one_hot_classes is not None:
            class_count = _validate_positive_int("one_hot_classes", one_hot_classes)
            self.arrays[-1] = [
                self._one_hot(label, class_count)
                for label in self.arrays[-1]
            ]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(f"ArrayDataset index {idx} out of range")
        return tuple(array[idx] for array in self.arrays)

    @staticmethod
    def _one_hot(label: Any, class_count: int) -> List[float]:
        if _is_sequence(label):
            if len(label) != 1:
                raise ValueError("one_hot_classes requires scalar class labels")
            label = label[0]

        class_index = int(label)
        if float(label) != float(class_index):
            raise ValueError("one_hot_classes requires integer class labels")
        if class_index < 0 or class_index >= class_count:
            raise ValueError(
                f"Class label {class_index} is out of bounds for one_hot_classes={class_count}"
            )

        encoded = [0.0] * class_count
        encoded[class_index] = 1.0
        return encoded


class CSVDataset(Dataset[tuple[List[float], List[float]]]):
    """Dataset wrapper for tabular CSV files.

    Args:
        filepath: Path to a CSV file.
        feature_cols: Feature column names or indexes. Defaults to all columns
            except target columns.
        target_cols: Target column names or indexes. Defaults to the last column.
        has_header: Whether the CSV contains a header row.
        one_hot_classes: Optional class count for one-hot target encoding.
    """

    def __init__(
        self,
        filepath: str,
        feature_cols: ColumnSelectors = None,
        target_cols: ColumnSelectors = None,
        has_header: bool = True,
        one_hot_classes: Optional[int] = None,
    ):
        if isinstance(has_header, bool) is False:
            raise TypeError("has_header must be a bool")

        with open(filepath, "r", newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))

        if not rows:
            raise ValueError(f"CSV file {filepath} is empty")

        header = rows[0] if has_header else None
        data_rows = rows[1:] if has_header else rows
        if not data_rows:
            raise ValueError(f"CSV file {filepath} does not contain data rows")

        column_count = len(header) if header is not None else len(data_rows[0])
        target_selectors = _as_column_list(target_cols, "target_cols")
        feature_selectors = _as_column_list(feature_cols, "feature_cols")

        target_indices = self._resolve_target_indices(
            target_selectors,
            header,
            column_count,
        )
        feature_indices = self._resolve_feature_indices(
            feature_selectors,
            target_indices,
            header,
            column_count,
        )

        features = []
        targets = []
        for row_number, row in enumerate(data_rows, start=2 if has_header else 1):
            if len(row) != column_count:
                raise ValueError(
                    f"CSV row {row_number} has {len(row)} columns, expected {column_count}"
                )

            features.append(self._read_float_columns(row, feature_indices, row_number))
            targets.append(self._read_float_columns(row, target_indices, row_number))

        self.dataset = ArrayDataset(features, targets, one_hot_classes=one_hot_classes)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[List[float], List[float]]:
        return self.dataset[idx]

    @staticmethod
    def _resolve_column(selector: ColumnSelector, header, column_count: int) -> int:
        if isinstance(selector, str):
            if header is None:
                raise ValueError("Cannot select CSV columns by name without a header row")
            try:
                column_index = header.index(selector)
            except ValueError as exc:
                raise ValueError(f"CSV column {selector!r} does not exist") from exc
        else:
            if isinstance(selector, bool):
                raise TypeError("CSV column selectors must be int or str")
            column_index = selector

        if column_index < 0:
            column_index += column_count
        if column_index < 0 or column_index >= column_count:
            raise ValueError(
                f"CSV column index {selector!r} is out of range for {column_count} columns"
            )
        return column_index

    @classmethod
    def _resolve_target_indices(cls, selectors, header, column_count: int) -> List[int]:
        if selectors is None:
            return [column_count - 1]
        return [cls._resolve_column(selector, header, column_count) for selector in selectors]

    @classmethod
    def _resolve_feature_indices(
        cls,
        selectors,
        target_indices: List[int],
        header,
        column_count: int,
    ) -> List[int]:
        if selectors is None:
            return [
                column_index
                for column_index in range(column_count)
                if column_index not in target_indices
            ]

        return [cls._resolve_column(selector, header, column_count) for selector in selectors]

    @staticmethod
    def _read_float_columns(row: List[str], column_indices: List[int], row_number: int) -> List[float]:
        values = []
        for column_index in column_indices:
            try:
                values.append(float(row[column_index]))
            except ValueError as exc:
                raise ValueError(
                    f"CSV row {row_number} column {column_index} is not a valid float"
                ) from exc
        return values


class SubsetDataset(Dataset[T_co]):
    """Dataset wrapping a subset of another dataset."""

    def __init__(self, dataset: Dataset[T_co], indices: List[int]):
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError("dataset must implement __len__ and __getitem__")

        dataset_length = len(dataset)
        self.dataset = dataset
        self.indices = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("SubsetDataset indices must be integers")
            if index < 0:
                index += dataset_length
            if index < 0 or index >= dataset_length:
                raise IndexError(f"SubsetDataset index {index} out of range")
            self.indices.append(index)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> T_co:
        if idx < 0:
            idx += len(self.indices)
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"SubsetDataset index {idx} out of range")
        return self.dataset[self.indices[idx]]


def random_split(
    dataset: Dataset[T_co],
    lengths: List[int],
    seed: Optional[int] = None,
) -> List[SubsetDataset[T_co]]:
    """Randomly split a dataset into non-overlapping subsets."""
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")

    dataset_length = len(dataset)
    if any(isinstance(length, bool) or not isinstance(length, int) for length in lengths):
        raise TypeError("lengths must contain integers")
    if any(length < 0 for length in lengths):
        raise ValueError("lengths must be non-negative")
    if sum(lengths) != dataset_length:
        raise ValueError("Sum of input lengths does not equal the length of the input dataset")

    indices = list(range(dataset_length))
    random_source = random.Random(seed)
    random_source.shuffle(indices)

    subsets = []
    start = 0
    for length in lengths:
        end = start + length
        subsets.append(SubsetDataset(dataset, indices[start:end]))
        start = end
    return subsets


def from_arrays(
    *arrays: Any,
    batch_size: int = 32,
    shuffle: bool = False,
    one_hot_classes: Optional[int] = None,
    drop_last: bool = False,
    seed: Optional[int] = None,
    dtype=None,
    device=None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Create a ready-to-use ``DataLoader`` from in-memory arrays."""
    dataset = ArrayDataset(*arrays, one_hot_classes=one_hot_classes)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
        dtype=dtype,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def from_csv(
    filepath: str,
    *,
    target: ColumnSelectors = None,
    features: ColumnSelectors = None,
    batch_size: int = 32,
    shuffle: bool = False,
    has_header: bool = True,
    one_hot_classes: Optional[int] = None,
    drop_last: bool = False,
    seed: Optional[int] = None,
    dtype=None,
    device=None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Create a ready-to-use ``DataLoader`` from a CSV file."""
    dataset = CSVDataset(
        filepath,
        feature_cols=features,
        target_cols=target,
        has_header=has_header,
        one_hot_classes=one_hot_classes,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
        dtype=dtype,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
