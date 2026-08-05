"""Advanced Orca training example using an explicit custom loop."""
import orca
import orca.nn as nn
import orca.optim as optim


def main():
    dataset = orca.data.ArrayDataset(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        [0, 1, 1, 0],
        one_hot_classes=2,
    )
    loader = orca.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        seed=42,
        dtype=orca.DType.FLOAT32,
        num_workers=2,
        prefetch_factor=2,
    )

    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    for epoch in range(10):
        epoch_loss = 0.0
        batches = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.to_list()[0]
            batches += 1

        print(f"epoch {epoch + 1}: loss={epoch_loss / batches:.4f}")


if __name__ == "__main__":
    main()
