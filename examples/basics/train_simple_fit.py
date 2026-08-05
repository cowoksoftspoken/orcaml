"""Beginner-friendly Orca training example using high-level defaults."""
import orca
import orca.nn as nn


def main():
    train_data = orca.data.from_arrays(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        [0, 1, 1, 0],
        batch_size=4,
        one_hot_classes=2,
    )

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    history = model.compile(
        optimizer="sgd",
        loss="crossentropy",
        lr=0.1,
        metrics=["accuracy"],
    ).fit(train_data, epochs=10)

    print(f"final loss: {history['loss'][-1]:.4f}")
    print(f"final accuracy: {history['accuracy'][-1]:.2f}")


if __name__ == "__main__":
    main()
