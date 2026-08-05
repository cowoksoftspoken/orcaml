"""Beginner Orca lifecycle example: train, evaluate, predict, save, load."""
from pathlib import Path
from tempfile import TemporaryDirectory

import orca
import orca.nn as nn


def build_model():
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def main():
    train_data = orca.data.from_arrays(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        [0, 1, 1, 0],
        batch_size=4,
        one_hot_classes=2,
    )

    model = build_model()
    with TemporaryDirectory() as checkpoint_dir:
        metrics_path = Path(checkpoint_dir) / "metrics.csv"
        history = model.compile(
            optimizer="sgd",
            loss="crossentropy",
            lr=0.1,
            metrics=["accuracy"],
        ).fit(
            train_data,
            epochs=10,
            verbose=1,
            callbacks=[orca.callbacks.CSVLogger(metrics_path)],
        )

        metrics = model.evaluate(train_data)
        predictions = model.predict([[1.0, 0.0]], batch_size=1)

        checkpoint_path = Path(checkpoint_dir) / "xor.safetensors"
        model.save(checkpoint_path)

        restored_model = build_model()
        restored_model.load(checkpoint_path)
        restored_predictions = restored_model.predict([[1.0, 0.0]], batch_size=1)
        metrics_header = metrics_path.read_text(encoding="utf-8").splitlines()[0]

    print(f"final loss: {history['loss'][-1]:.4f}")
    print(f"eval accuracy: {metrics['accuracy']:.2f}")
    print(f"metrics file header: {metrics_header}")
    print(f"prediction: {predictions.to_list()}")
    print(f"restored prediction: {restored_predictions.to_list()}")


if __name__ == "__main__":
    main()
