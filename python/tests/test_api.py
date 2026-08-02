import pytest
import orca
import orca.nn as nn
import orca.optim as optim

@pytest.fixture
def cpu_device():
    """Fixture to provide CPU device string."""
    return "cpu"

@pytest.fixture
def gpu_device():
    """Fixture to provide GPU device string."""
    return "gpu"

@pytest.fixture
def simple_model():
    """Fixture to construct a simple 2->2 Linear model."""
    return nn.Sequential(nn.Linear(2, 2))

@pytest.mark.parametrize("shape,expected_len", [
    ([2, 3], 6),
    ([1, 4], 4),
    ([5], 5),
])
def test_tensor_creation(shape, expected_len):
    """Verify tensor allocation shape and zero initialization using pytest parametrization."""
    x = orca.Tensor.zeros(shape)
    assert x.shape == shape
    assert len(x.to_list()) == expected_len
    assert all(val == 0.0 for val in x.to_list())

@pytest.mark.parametrize("a_val,b_val,op", [
    ([1.0, 2.0], [3.0, 4.0], "add"),
    ([2.0, 3.0], [4.0, 5.0], "sub"),
    ([1.0, 2.0], [3.0, 4.0], "mul"),
])
def test_tensor_ops(a_val, b_val, op):
    """Test elementary binary arithmetic operations on parameterized inputs."""
    a = orca.Tensor.from_list(a_val, shape=[1, len(a_val)])
    b = orca.Tensor.from_list(b_val, shape=[1, len(b_val)])
    
    if op == "add":
        c = a + b
        assert c.to_list() == [x + y for x, y in zip(a_val, b_val)]
    elif op == "sub":
        c = a - b
        assert c.to_list() == [x - y for x, y in zip(a_val, b_val)]
    elif op == "mul":
        c = a * b
        assert c.to_list() == [x * y for x, y in zip(a_val, b_val)]

def test_autograd_basic():
    """Verify standard automatic differentiation backward gradient accumulation."""
    x = orca.Tensor.from_list([2.0], shape=[1, 1], requires_grad=True)
    y = x * x  # y = x^2
    y.backward()
    
    assert x.grad() is not None
    # dy/dx = 2 * x = 4.0
    assert abs(x.grad().to_list()[0] - 4.0) < 1e-5

def test_shape_mismatch_exception():
    """Verify that backend shape mismatches raise a ValueError using pytest.raises."""
    a = orca.Tensor.from_list([1.0, 2.0], shape=[1, 2])
    b = orca.Tensor.from_list([1.0, 2.0, 3.0], shape=[1, 3])
    with pytest.raises(ValueError):
        _ = a + b

def test_dtype_mismatch_exception():
    """Verify that backend dtype mismatches raise a ValueError."""
    a = orca.Tensor.zeros([1, 2], dtype=orca.DType.FLOAT32)
    b = orca.Tensor.zeros([1, 2], dtype=orca.DType.FLOAT16)
    with pytest.raises(ValueError):
        _ = a + b

def test_linear_layer(simple_model):
    """Test linear forward pass with a shared model fixture."""
    x = orca.Tensor.ones([1, 2])
    out = simple_model(x)
    assert out.shape == [1, 2]

def test_crossentropy_loss():
    """Test CrossEntropyLoss stability and output range."""
    loss_fn = nn.CrossEntropyLoss()
    pred = orca.Tensor.from_list([1.0, 0.0], shape=[1, 2])
    target = orca.Tensor.from_list([1.0, 0.0], shape=[1, 2])
    loss = loss_fn(pred, target)
    assert loss.to_list()[0] > 0.0

@pytest.mark.parametrize("optimizer_class", [
    optim.SGD,
    optim.Adam,
    optim.AdamW,
])
def test_optimizers(simple_model, optimizer_class):
    """Parametrized verification of SGD, Adam, and AdamW optimizer parameter updates."""
    if optimizer_class == optim.SGD:
        optimizer = optimizer_class(simple_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01)
    elif optimizer_class == optim.AdamW:
        optimizer = optimizer_class(simple_model.parameters(), lr=0.01, weight_decay=0.01)
    else:
        optimizer = optimizer_class(simple_model.parameters(), lr=0.01)
        
    x = orca.Tensor.ones([1, 2])
    y = orca.Tensor.from_list([1.0, 0.0], shape=[1, 2])
    loss_fn = nn.MSELoss()
    
    # Save initial weights
    params = list(simple_model.parameters())
    init_weights = params[0].tensor.to_list()
    
    # Update
    optimizer.zero_grad()
    pred = simple_model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    
    # Verify weights actually updated
    updated_weights = params[0].tensor.to_list()
    assert init_weights != updated_weights


def test_optimizer_rejects_invalid_parameter_lists():
    """Verify optimizers fail fast on empty or non-Parameter inputs."""
    with pytest.raises(ValueError):
        optim.SGD([])

    with pytest.raises(TypeError):
        optim.SGD([orca.Tensor.ones([1])])


@pytest.mark.parametrize("optimizer_class", [optim.Adam, optim.AdamW])
def test_adam_family_state_matches_parameter_dtype(optimizer_class):
    """Verify Adam-family state tensors preserve the parameter dtype."""
    parameter = nn.Parameter(
        orca.Tensor.ones([2], dtype=orca.DType.FLOAT16, requires_grad=True)
    )

    optimizer = optimizer_class([parameter])

    assert optimizer.m[0].dtype == parameter.tensor.dtype
    assert optimizer.v[0].dtype == parameter.tensor.dtype


@pytest.mark.parametrize("optimizer_class", [optim.Adam, optim.AdamW])
def test_adam_family_no_grad_step_keeps_step_counter(optimizer_class):
    """Verify Adam-family step counters only advance when a gradient is applied."""
    parameter = nn.Parameter(orca.Tensor.ones([1], requires_grad=True))
    optimizer = optimizer_class([parameter])

    optimizer.step()

    assert optimizer.t == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda parameter: optim.SGD([parameter], lr=-0.1),
        lambda parameter: optim.SGD([parameter], momentum=-0.1),
        lambda parameter: optim.SGD([parameter], dampening=1.1),
        lambda parameter: optim.Adam([parameter], betas=(0.9, 1.0)),
        lambda parameter: optim.Adam([parameter], eps=0.0),
        lambda parameter: optim.AdamW([parameter], weight_decay=-0.1),
    ],
)
def test_optimizers_reject_invalid_hyperparameters(factory):
    """Verify optimizer hyperparameter errors are explicit."""
    parameter = nn.Parameter(orca.Tensor.ones([1], requires_grad=True))

    with pytest.raises((TypeError, ValueError)):
        factory(parameter)


@pytest.mark.parametrize(
    "factory",
    [
        lambda optimizer: optim.StepLR(optimizer, step_size=0),
        lambda optimizer: optim.StepLR(optimizer, step_size=1, gamma=-0.1),
        lambda optimizer: optim.CosineAnnealingLR(optimizer, T_max=0),
        lambda optimizer: optim.CosineAnnealingLR(optimizer, T_max=1, eta_min=-0.1),
        lambda optimizer: optim.LinearWarmup(optimizer, warmup_epochs=0),
    ],
)
def test_lr_schedulers_reject_invalid_hyperparameters(factory):
    """Verify scheduler hyperparameter errors are explicit."""
    parameter = nn.Parameter(orca.Tensor.ones([1], requires_grad=True))
    optimizer = optim.SGD([parameter])

    with pytest.raises((TypeError, ValueError)):
        factory(optimizer)


def test_numerical_gradients():
    """Check analytical autograd correctness using numerical finite differences."""
    x_val = 2.0
    eps = 1e-4

    # Analytical
    x = orca.Tensor.from_list([x_val], shape=[1], requires_grad=True)
    y = x * x * x + x * x * orca.Tensor.scalar(2.0) + x * orca.Tensor.scalar(5.0)
    y.backward()
    grad_analytical = x.grad().to_list()[0]

    # Numerical f(x + eps)
    x_plus = orca.Tensor.from_list([x_val + eps], shape=[1])
    y_plus = x_plus * x_plus * x_plus + x_plus * x_plus * orca.Tensor.scalar(2.0) + x_plus * orca.Tensor.scalar(5.0)
    
    # f(x - eps)
    x_minus = orca.Tensor.from_list([x_val - eps], shape=[1])
    y_minus = x_minus * x_minus * x_minus + x_minus * x_minus * orca.Tensor.scalar(2.0) + x_minus * orca.Tensor.scalar(5.0)

    grad_numerical = (y_plus.to_list()[0] - y_minus.to_list()[0]) / (2.0 * eps)

    assert abs(grad_analytical - 25.0) < 1e-2
    assert abs(grad_analytical - grad_numerical) < 1e-2


@pytest.mark.parametrize("scheduler_class,kwargs", [
    (optim.StepLR, {"step_size": 2, "gamma": 0.5}),
    (optim.CosineAnnealingLR, {"T_max": 10, "eta_min": 0.001}),
    (optim.LinearWarmup, {"warmup_epochs": 5}),
])
def test_lr_schedulers(simple_model, scheduler_class, kwargs):
    """Verify learning rate schedulers update optimizer learning rates correctly."""
    optimizer = optim.SGD(simple_model.parameters(), lr=0.1)
    scheduler = scheduler_class(optimizer, **kwargs)
    
    initial_lr = optimizer.lr
    
    # Step the scheduler
    scheduler.step()
    first_step_lr = optimizer.lr
    
    if scheduler_class == optim.StepLR:
        # Step 1: last_epoch=1, step_size=2 -> no change
        assert first_step_lr == initial_lr
        scheduler.step() # last_epoch=2 -> lr *= gamma
        assert optimizer.lr == initial_lr * 0.5
    elif scheduler_class == optim.CosineAnnealingLR:
        # Should change because of cosine decay
        assert first_step_lr < initial_lr
    elif scheduler_class == optim.LinearWarmup:
        # At epoch 0 (initially) lr is 0.1, at first step (last_epoch=1) it should be 0.1 * 1/5 = 0.02
        assert first_step_lr == pytest.approx(0.02)


def test_linear_numerical_gradient():
    """Verify Linear layer gradients using numerical finite differences."""
    # Simple model: y = x @ W
    # where x is [1, 2], W is [2, 1]. Out is [1, 1].
    x_val = [1.5, -2.0]
    w_val = [0.8, 1.2]
    eps = 1e-4
    
    # Analytical backward pass
    x = orca.Tensor.from_list(x_val, shape=[1, 2])
    W = orca.Tensor.from_list(w_val, shape=[2, 1], requires_grad=True)
    
    y = x @ W
    y.backward()
    grad_W = W.grad().to_list()
    
    # Numerical backward pass
    # f(W + eps) for W[0]
    W_plus = orca.Tensor.from_list([w_val[0] + eps, w_val[1]], shape=[2, 1])
    y_plus = (x @ W_plus).to_list()[0]
    
    # f(W - eps) for W[0]
    W_minus = orca.Tensor.from_list([w_val[0] - eps, w_val[1]], shape=[2, 1])
    y_minus = (x @ W_minus).to_list()[0]
    
    grad_num_0 = (y_plus - y_minus) / (2.0 * eps)
    
    # Compare
    assert abs(grad_W[0] - grad_num_0) < 1e-2
    assert abs(grad_W[0] - x_val[0]) < 1e-3


def test_grad_scaler(simple_model):
    """Verify GradScaler scales loss, steps optimizer, and updates scales correctly."""
    from orca import GradScaler
    scaler = GradScaler(init_scale=1024.0, growth_interval=2)
    optimizer = optim.SGD(simple_model.parameters(), lr=0.01)
    
    x = orca.Tensor.ones([1, 2])
    y = orca.Tensor.from_list([1.0, 0.0], shape=[1, 2])
    loss_fn = nn.MSELoss()
    
    # 1. Scale loss
    pred = simple_model(x)
    loss = loss_fn(pred, y)
    scaled_loss = scaler.scale_loss(loss)
    
    assert scaled_loss.to_list()[0] == pytest.approx(loss.to_list()[0] * 1024.0)
    
    # 2. Backward
    optimizer.zero_grad()
    scaled_loss.backward()
    
    # 3. Step
    scaler.step(optimizer)
    scaler.update()
    
    # Scale should remain same since growth interval is 2
    assert scaler.scale == 1024.0
    
    # Step again to trigger growth
    pred = simple_model(x)
    loss = loss_fn(pred, y)
    scaled_loss = scaler.scale_loss(loss)
    optimizer.zero_grad()
    scaled_loss.backward()
    scaler.step(optimizer)
    scaler.update()
    
    # Scale should have grown by growth_factor (2.0)
    assert scaler.scale == 2048.0


def test_tensor_chunk():
    """Verify Tensor.chunk splits a tensor correctly and gradients flow back through chunks."""
    x = orca.Tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], requires_grad=True)
    
    # Chunk along last dimension (dim=1) into 3 chunks
    chunks = x.chunk(3, dim=-1)
    assert len(chunks) == 3
    assert chunks[0].shape == [2, 1]
    assert chunks[1].shape == [2, 1]
    assert chunks[2].shape == [2, 1]
    
    assert chunks[0].to_list() == [1.0, 4.0]
    assert chunks[1].to_list() == [2.0, 5.0]
    assert chunks[2].to_list() == [3.0, 6.0]
    
    # Verify backward pass through chunks
    loss = chunks[0] * 10.0 + chunks[1] * 20.0 + chunks[2] * 30.0
    # Sum to scalar
    loss_sum = loss.sum_to_shape([1])
    loss_sum.backward()
    
    # Expected grads:
    # chunk 0: 10, chunk 1: 20, chunk 2: 30
    assert x.grad().to_list() == [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]


def test_multi_head_attention_packed():
    """Verify MultiHeadAttention works correctly for both self-attention (packed) and cross-attention."""
    mha = nn.MultiHeadAttention(embed_dim=4, num_heads=2)
    
    # Self-attention (same query, key, value)
    x = orca.Tensor.ones([2, 3, 4], requires_grad=True)
    out_self = mha(x, x, x)
    assert out_self.shape == [2, 3, 4]
    
    # Backward pass
    loss_self = out_self.sum_to_shape([1])
    loss_self.backward()
    assert x.grad() is not None
    
    # Cross-attention (different inputs)
    q = orca.Tensor.ones([2, 3, 4], requires_grad=True)
    kv = orca.Tensor.ones([2, 5, 4], requires_grad=True)
    out_cross = mha(q, kv, kv)
    assert out_cross.shape == [2, 3, 4]
    
    loss_cross = out_cross.sum_to_shape([1])
    loss_cross.backward()
    assert q.grad() is not None
    assert kv.grad() is not None


def test_parameter_initializers():
    """Verify parameter initialization helpers (Kaiming, Xavier, normal, uniform) run correctly."""
    from orca.nn import init
    
    param = nn.Parameter(orca.Tensor.zeros([10, 20]))
    
    # 1. Kaiming Normal
    init.kaiming_normal_(param)
    assert any(val != 0.0 for val in param.tensor.to_list())
    
    # 2. Xavier Uniform
    init.xavier_uniform_(param)
    assert any(val != 0.0 for val in param.tensor.to_list())
    
    # 3. Zeros
    init.zeros_(param)
    assert all(val == 0.0 for val in param.tensor.to_list())
    
    # 4. Ones
    init.ones_(param)
    assert all(val == 1.0 for val in param.tensor.to_list())


def test_high_level_dataset_wrappers(tmp_path):
    """Verify ArrayDataset, CSVDataset, and random_split load and split data correctly."""
    from orca.data import ArrayDataset, CSVDataset, DataLoader, random_split
    import numpy as np
    
    # 1. ArrayDataset with list & numpy inputs
    X_np = np.random.randn(10, 4)
    y_np = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    
    # Enable automatic one-hot encoding to 3 classes
    dataset = ArrayDataset(X_np, y_np, one_hot_classes=3)
    assert len(dataset) == 10
    
    x_sample, y_sample = dataset[0]
    assert len(x_sample) == 4
    assert y_sample == [1.0, 0.0, 0.0]  # Class 0 one-hot encoded
    
    # 2. random_split check
    train_set, val_set = random_split(dataset, [8, 2])
    assert len(train_set) == 8
    assert len(val_set) == 2
    
    # 3. CSVDataset check
    csv_file = tmp_path / "data.csv"
    with open(csv_file, "w") as f:
        f.write("feat1,feat2,target\n")
        f.write("1.0,2.0,0\n")
        f.write("3.0,4.0,1\n")
        f.write("5.0,6.0,2\n")
        f.write("7.0,8.0,0\n")
        
    # Load using column names
    csv_dataset = CSVDataset(
        str(csv_file),
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        has_header=True,
        one_hot_classes=3
    )
    assert len(csv_dataset) == 4
    x_c, y_c = csv_dataset[1]
    assert x_c == [3.0, 4.0]
    assert y_c == [0.0, 1.0, 0.0]  # Class 1 one-hot encoded
    
    # 4. DataLoader run
    loader = DataLoader(csv_dataset, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    
    X_batch, Y_batch = batches[0]
    assert X_batch.shape == [2, 2]
    assert Y_batch.shape == [2, 3]



