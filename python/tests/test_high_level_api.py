import time

import pytest

import orca
import orca.nn as nn
from orca import data


def test_from_arrays_beginner_path_batches_to_tensors():
    loader = data.from_arrays(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [0, 1, 2],
        batch_size=2,
        one_hot_classes=3,
    )

    first_batch = next(iter(loader))
    batch_inputs, batch_targets = first_batch

    assert batch_inputs.shape == [2, 2]
    assert batch_targets.shape == [2, 3]
    assert batch_targets.to_list() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_dataloader_advanced_options_are_explicit():
    dataset = data.ArrayDataset(
        [[1.0], [2.0], [3.0]],
        [[1.0], [2.0], [3.0]],
    )
    loader = data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        seed=7,
        dtype=orca.DType.FLOAT16,
    )

    batches = list(loader)

    assert len(loader) == 1
    assert len(batches) == 1
    assert batches[0][0].dtype == orca.DType.FLOAT16
    assert batches[0][1].dtype == orca.DType.FLOAT16


def test_dataloader_rejects_ragged_batches():
    dataset = data.ArrayDataset(
        [[1.0], [2.0, 3.0]],
        [[1.0], [2.0]],
    )
    loader = data.DataLoader(dataset, batch_size=2)

    with pytest.raises(ValueError, match="same shape"):
        list(loader)


def test_dataloader_prefetch_matches_sequential_order():
    dataset = data.ArrayDataset(
        [[0.0], [1.0], [2.0], [3.0], [4.0]],
        [[10.0], [11.0], [12.0], [13.0], [14.0]],
    )
    sequential_loader = data.DataLoader(dataset, batch_size=2, num_workers=0)
    prefetched_loader = data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        prefetch_factor=2,
    )

    sequential_batches = [
        (batch_inputs.to_list(), batch_targets.to_list())
        for batch_inputs, batch_targets in sequential_loader
    ]
    prefetched_batches = [
        (batch_inputs.to_list(), batch_targets.to_list())
        for batch_inputs, batch_targets in prefetched_loader
    ]

    assert prefetched_batches == sequential_batches


def test_dataloader_prefetch_preserves_order_when_workers_finish_out_of_order():
    class DelayedDataset(data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            time.sleep((4 - idx) * 0.005)
            return [float(idx)]

    loader = data.DataLoader(
        DelayedDataset(),
        batch_size=1,
        num_workers=4,
        prefetch_factor=1,
    )

    assert [batch.to_list()[0] for batch in loader] == [0.0, 1.0, 2.0, 3.0]


def test_dataloader_prefetch_propagates_dataset_errors():
    class FailingDataset(data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            if idx == 1:
                raise ValueError("bad sample")
            return [float(idx)]

    loader = data.DataLoader(FailingDataset(), batch_size=1, num_workers=2)

    with pytest.raises(ValueError, match="bad sample") as error_info:
        list(loader)

    assert (
        "DataLoader failed while loading dataset index 1"
        in getattr(error_info.value, "__notes__", [])
    )


def test_dataloader_rejects_invalid_prefetch_options():
    dataset = data.ArrayDataset([0, 1])

    with pytest.raises(TypeError, match="num_workers"):
        data.DataLoader(dataset, num_workers=True)
    with pytest.raises(ValueError, match="num_workers"):
        data.DataLoader(dataset, num_workers=-1)
    with pytest.raises(ValueError, match="prefetch_factor"):
        data.DataLoader(dataset, num_workers=1, prefetch_factor=0)


def test_from_csv_defaults_to_last_column(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "x1,x2,label\n"
        "1.0,2.0,0\n"
        "3.0,4.0,1\n",
        encoding="utf-8",
    )

    loader = data.from_csv(
        str(csv_path),
        batch_size=2,
        one_hot_classes=2,
    )
    batch_inputs, batch_targets = next(iter(loader))

    assert batch_inputs.shape == [2, 2]
    assert batch_inputs.to_list() == [1.0, 2.0, 3.0, 4.0]
    assert batch_targets.shape == [2, 2]
    assert batch_targets.to_list() == [1.0, 0.0, 0.0, 1.0]


def test_csv_dataset_rejects_missing_named_column(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("x,label\n1.0,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not exist"):
        data.CSVDataset(str(csv_path), feature_cols="missing", target_cols="label")


def test_random_split_seed_is_deterministic():
    dataset = data.ArrayDataset([0, 1, 2, 3])

    first_split = data.random_split(dataset, [2, 2], seed=42)
    second_split = data.random_split(dataset, [2, 2], seed=42)

    assert [subset.indices for subset in first_split] == [
        subset.indices for subset in second_split
    ]


def test_model_compile_fit_from_arrays():
    model = nn.Sequential(nn.Linear(2, 1))

    history = model.compile(optimizer="sgd", loss="mse", lr=0.01).fit(
        [[0.0, 0.0], [1.0, 1.0]],
        [[0.0], [1.0]],
        epochs=2,
        batch_size=2,
    )

    assert list(history.keys()) == ["loss"]
    assert len(history["loss"]) == 2
    assert all(loss_value >= 0.0 for loss_value in history["loss"])


def test_model_fit_accepts_loader_and_reports_accuracy():
    loader = data.from_arrays(
        [[1.0, 0.0], [0.0, 1.0]],
        [0, 1],
        batch_size=2,
        one_hot_classes=2,
    )
    model = nn.Sequential(nn.Linear(2, 2))

    history = model.compile(
        optimizer="sgd",
        loss="crossentropy",
        lr=0.01,
        metrics=["accuracy"],
    ).fit(loader, epochs=1)

    assert len(history["loss"]) == 1
    assert len(history["accuracy"]) == 1
    assert 0.0 <= history["accuracy"][0] <= 1.0


def test_model_fit_arrays_support_prefetch_workers():
    model = nn.Sequential(nn.Linear(2, 1))

    history = model.compile(optimizer="sgd", loss="mse", lr=0.01).fit(
        [[0.0, 0.0], [1.0, 1.0]],
        [[0.0], [1.0]],
        epochs=1,
        batch_size=1,
        num_workers=2,
        prefetch_factor=1,
    )

    assert len(history["loss"]) == 1


def test_model_fit_rejects_loader_worker_overrides():
    loader = data.from_arrays(
        [[0.0, 0.0], [1.0, 1.0]],
        [[0.0], [1.0]],
        batch_size=1,
        num_workers=1,
    )
    model = nn.Sequential(nn.Linear(2, 1)).compile(optimizer="sgd", loss="mse")

    with pytest.raises(ValueError, match="model creates the DataLoader"):
        model.fit(loader, epochs=1, num_workers=1)


def test_model_fit_validation_history_and_restores_mode():
    train_loader = data.from_arrays(
        [[1.0, 0.0], [0.0, 1.0]],
        [0, 1],
        batch_size=2,
        one_hot_classes=2,
    )
    validation_loader = data.from_arrays(
        [[1.0, 0.0], [0.0, 1.0]],
        [0, 1],
        batch_size=2,
        one_hot_classes=2,
    )
    model = nn.Sequential(nn.Linear(2, 2))
    model.eval()

    history = model.compile(
        optimizer="sgd",
        loss="crossentropy",
        lr=0.01,
        metrics="accuracy",
    ).fit(train_loader, epochs=2, validation_data=validation_loader)

    assert list(history.keys()) == ["loss", "accuracy", "val_loss", "val_accuracy"]
    assert len(history["loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert model.training is False


def test_model_fit_verbose_logs_epoch_metrics(capsys):
    loader = data.from_arrays(
        [[1.0, 0.0], [0.0, 1.0]],
        [0, 1],
        batch_size=2,
        one_hot_classes=2,
    )
    model = nn.Sequential(nn.Linear(2, 2))

    model.compile(
        optimizer="sgd",
        loss="crossentropy",
        lr=0.01,
        metrics="accuracy",
    ).fit(loader, epochs=1, verbose=1)

    captured = capsys.readouterr()
    assert "Epoch 1/1" in captured.out
    assert "loss=" in captured.out
    assert "accuracy=" in captured.out


def test_model_fit_callback_receives_logs_and_stops_training():
    loader = data.from_arrays(
        [[1.0, 0.0], [0.0, 1.0]],
        [0, 1],
        batch_size=2,
        one_hot_classes=2,
    )
    events = []

    class StopAfterFirstEpoch(orca.callbacks.Callback):
        def on_train_begin(self, logs=None):
            events.append(("begin", logs["epochs"]))

        def on_epoch_end(self, epoch, logs=None):
            events.append(("epoch", epoch, sorted(logs)))
            self.model.stop_training = True

        def on_train_end(self, logs=None):
            events.append(("end", len(logs["history"]["loss"])))

    model = nn.Sequential(nn.Linear(2, 2))
    history = model.compile(
        optimizer="sgd",
        loss="crossentropy",
        lr=0.01,
        metrics="accuracy",
    ).fit(
        loader,
        epochs=3,
        validation_data=loader,
        callbacks=[StopAfterFirstEpoch()],
    )

    assert len(history["loss"]) == 1
    assert events[0] == ("begin", 3)
    assert events[1] == ("epoch", 0, ["accuracy", "loss", "val_accuracy", "val_loss"])
    assert events[2] == ("end", 1)


def test_model_fit_csv_logger_writes_and_appends_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    loader = data.from_arrays(
        [[1.0, 0.0], [0.0, 1.0]],
        [0, 1],
        batch_size=2,
        one_hot_classes=2,
    )

    model = nn.Sequential(nn.Linear(2, 2))
    model.compile(
        optimizer="sgd",
        loss="crossentropy",
        lr=0.01,
        metrics="accuracy",
    ).fit(loader, epochs=2, callbacks=[orca.callbacks.CSVLogger(metrics_path)])

    restored_model = nn.Sequential(nn.Linear(2, 2))
    restored_model.compile(
        optimizer="sgd",
        loss="crossentropy",
        lr=0.01,
        metrics="accuracy",
    ).fit(
        loader,
        epochs=1,
        callbacks=[orca.callbacks.CSVLogger(metrics_path, append=True)],
    )

    rows = metrics_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].split(",") == ["epoch", "loss", "accuracy"]
    assert len(rows) == 4
    assert rows[1].startswith("1,")
    assert rows[2].startswith("2,")
    assert rows[3].startswith("1,")


def test_model_fit_rejects_invalid_callbacks():
    model = nn.Sequential(nn.Linear(2, 1)).compile(optimizer="sgd", loss="mse")

    with pytest.raises(TypeError, match="callbacks"):
        model.fit(
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0], [1.0]],
            callbacks=object(),
        )


def test_model_evaluate_reports_metrics_and_restores_mode():
    loader = data.from_arrays(
        [[1.0, 0.0], [0.0, 1.0]],
        [0, 1],
        batch_size=2,
        one_hot_classes=2,
    )
    model = nn.Sequential(nn.Linear(2, 2)).compile(
        optimizer="sgd",
        loss="crossentropy",
        metrics=["accuracy"],
    )

    model.train()
    train_mode_result = model.evaluate(loader)
    assert list(train_mode_result.keys()) == ["loss", "accuracy"]
    assert 0.0 <= train_mode_result["accuracy"] <= 1.0
    assert model.training is True

    model.eval()
    eval_mode_result = model.evaluate(loader)
    assert list(eval_mode_result.keys()) == ["loss", "accuracy"]
    assert model.training is False


def test_model_predict_supports_arrays_loader_and_restores_mode():
    loader = data.from_arrays(
        [[1.0, 0.0], [0.0, 1.0]],
        [0, 1],
        batch_size=2,
        one_hot_classes=2,
    )
    model = nn.Sequential(nn.Linear(2, 2))

    model.train()
    loader_predictions = model.predict(loader)
    assert loader_predictions.shape == [2, 2]
    assert loader_predictions.requires_grad is False
    assert model.training is True

    model.eval()
    array_predictions = model.predict([[1.0, 0.0], [0.0, 1.0]], batch_size=2)
    assert array_predictions.shape == [2, 2]
    assert array_predictions.requires_grad is False
    assert model.training is False


def test_model_predict_multiple_batches_return_list():
    model = nn.Sequential(nn.Linear(2, 1))

    predictions = model.predict(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        batch_size=1,
    )

    assert isinstance(predictions, list)
    assert [prediction.shape for prediction in predictions] == [[1, 1], [1, 1], [1, 1]]


def test_model_save_load_roundtrip(tmp_path):
    checkpoint_path = tmp_path / "model.safetensors"
    model = nn.Sequential(nn.Linear(2, 1))
    restored_model = nn.Sequential(nn.Linear(2, 1))

    assert model.save(checkpoint_path) is model
    assert restored_model.load(checkpoint_path) is restored_model
    assert restored_model.state_dict() == model.state_dict()


def test_model_load_missing_checkpoint_fails_clearly(tmp_path):
    model = nn.Sequential(nn.Linear(2, 1))

    with pytest.raises(FileNotFoundError, match="checkpoint file not found"):
        model.load(tmp_path / "missing.safetensors")


def test_model_fit_requires_compile():
    model = nn.Sequential(nn.Linear(2, 1))

    with pytest.raises(RuntimeError, match="compile"):
        model.fit([[0.0, 0.0]], [[0.0]])

    with pytest.raises(RuntimeError, match="compile"):
        model.evaluate([[0.0, 0.0]], [[0.0]])


def test_model_compile_rejects_unknown_names():
    model = nn.Sequential(nn.Linear(2, 1))

    with pytest.raises(ValueError, match="Unknown optimizer"):
        model.compile(optimizer="unknown", loss="mse")

    with pytest.raises(ValueError, match="Unknown loss"):
        model.compile(optimizer="sgd", loss="unknown")
