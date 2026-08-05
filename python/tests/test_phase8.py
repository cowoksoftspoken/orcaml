import pytest
import socket
import struct
import orca
import orca.nn as nn
import orca.optim as optim
from orca.distributed import DistributedDataParallel, _recv_exact, _recv_gradient
from orca.autocast import autocast, GradScaler

def test_autocast_context():
    assert not autocast._enabled
    with autocast(dtype=orca.DType.FLOAT16) as ac:
        assert autocast._enabled
        assert autocast._dtype == orca.DType.FLOAT16
    assert not autocast._enabled

def test_grad_scaler():
    scaler = GradScaler(init_scale=1024.0)
    loss = orca.Tensor.scalar(2.5, requires_grad=True)
    scaled_loss = scaler.scale_loss(loss)
    assert scaled_loss.to_list()[0] == 2560.0
    
    model = nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    x = orca.Tensor.randn([1, 2])
    out = model(x)
    loss = out.sum()
    
    scaled_loss = scaler.scale_loss(loss)
    scaled_loss.backward()
    
    scaler.step(optimizer)
    scaler.update()

    half_loss = orca.Tensor.scalar(2.5, dtype=orca.DType.FLOAT16, requires_grad=True)
    assert scaler.scale_loss(half_loss).dtype == orca.DType.FLOAT16


def test_autocast_preserves_parameter_gradients():
    model = nn.Linear(2, 2)
    inputs = orca.Tensor.ones([1, 2], requires_grad=True)

    with autocast(dtype=orca.DType.FLOAT16) as context:
        assert context is not None
        output = model(inputs)

    output.sum().backward()
    assert model.weight.tensor.grad() is not None
    assert model.bias.tensor.grad() is not None


def test_factory_functions_preserve_requested_dtype():
    for dtype in (orca.DType.FLOAT16, orca.DType.BFLOAT16, orca.DType.FLOAT64):
        assert orca.Tensor.ones([2], dtype=dtype).dtype == dtype
        assert orca.Tensor.scalar(1.0, dtype=dtype).dtype == dtype
        assert orca.Tensor.randn([2], dtype=dtype).dtype == dtype


def test_ddp_rejects_invalid_configuration():
    model = nn.Linear(1, 1)

    with pytest.raises(ValueError, match="world_size must be positive"):
        DistributedDataParallel(model, rank=0, world_size=0)

    with pytest.raises(ValueError, match=r"rank must be in \[0, 1\)"):
        DistributedDataParallel(model, rank=1, world_size=1)

    with pytest.raises(ValueError, match="master_addr must use"):
        DistributedDataParallel(model, master_addr="127.0.0.1")

    with pytest.raises(ValueError, match="timeout must be positive"):
        DistributedDataParallel(model, timeout=0.0)


@pytest.mark.skipif(
    not hasattr(socket, "socketpair"),
    reason="socketpair is required for local protocol tests",
)
def test_ddp_recv_exact_rejects_short_frames():
    receiver, sender = socket.socketpair()
    try:
        sender.sendall(b"abc")
        sender.close()

        with pytest.raises(RuntimeError, match="closed connection"):
            _recv_exact(receiver, 4)
    finally:
        receiver.close()


@pytest.mark.skipif(
    not hasattr(socket, "socketpair"),
    reason="socketpair is required for local protocol tests",
)
def test_ddp_rejects_gradient_size_mismatch():
    receiver, sender = socket.socketpair()
    try:
        sender.sendall(struct.pack("!Q", 2))
        sender.sendall(struct.pack("!2f", 1.0, 2.0))

        with pytest.raises(RuntimeError, match="expected 3 elements, got 2"):
            _recv_gradient(receiver, expected_elements=3)
    finally:
        receiver.close()
        sender.close()


def run_ddp_worker(rank, world_size, master_addr, x_data, result_queue):
    import orca
    import orca.nn as nn
    from orca.distributed import DistributedDataParallel
    
    base_model = nn.Linear(3, 3)
    base_model.weight.tensor = orca.Tensor.from_list([0.1]*9, shape=[3, 3], requires_grad=True)
    base_model.bias.tensor = orca.Tensor.from_list([0.1]*3, shape=[1, 3], requires_grad=True)
    
    ddp_model = DistributedDataParallel(base_model, rank=rank, world_size=world_size, master_addr=master_addr)
    x = orca.Tensor.from_list(x_data, shape=[2, 3])
    out = ddp_model(x)
    loss = out.sum()
    loss.backward()
    
    grads_before = [p.tensor.grad().to_list() for p in ddp_model.parameters() if p.tensor.grad() is not None]
    try:
        ddp_model.all_reduce_gradients()
        grads_after = [p.tensor.grad().to_list() for p in ddp_model.parameters() if p.tensor.grad() is not None]
        result_queue.put((rank, grads_before, grads_after))
    finally:
        ddp_model.close()

def test_ddp():
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    master_addr = "127.0.0.1:19888"
    x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    
    p_worker = ctx.Process(
        target=run_ddp_worker,
        args=(1, 2, master_addr, x_data, result_queue)
    )
    p_worker.start()
    
    p_master = ctx.Process(
        target=run_ddp_worker,
        args=(0, 2, master_addr, x_data, result_queue)
    )
    p_master.start()
    
    p_master.join()
    p_worker.join()
    
    res_0 = result_queue.get()
    res_1 = result_queue.get()
    
    results = {res_0[0]: res_0, res_1[0]: res_1}
    
    grads_before_0 = results[0][1]
    grads_after_0 = results[0][2]
    
    for gb, ga in zip(grads_before_0, grads_after_0):
        for b_val, a_val in zip(gb, ga):
            assert a_val == pytest.approx(b_val, abs=1e-5)

def test_quantized_linear():
    model = nn.Linear(4, 2)
    model.weight.tensor = orca.Tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape=[4, 2])
    model.bias.tensor = orca.Tensor.from_list([0.1, 0.2], shape=[1, 2])
    
    q_model = nn.QuantizedLinear.from_float(model)
    assert q_model.quantized
    assert len(q_model.weight_q) == 8
    
    x = orca.Tensor.from_list([0.5, 1.0, -0.5, -1.0], shape=[1, 4])
    y_float = model(x)
    y_quant = q_model(x)
    
    f_list = y_float.to_list()
    q_list = y_quant.to_list()
    for f, q in zip(f_list, q_list):
        assert f == pytest.approx(q, abs=0.2)

def test_autocast_linear():
    model = nn.Linear(4, 2)
    x = orca.Tensor.randn([2, 4])
    
    # Run under float16 autocast context
    with autocast(dtype=orca.DType.FLOAT16):
        out = model(x)
        assert out.dtype == orca.DType.FLOAT16
        
    # Run under bfloat16 autocast context
    with autocast(dtype=orca.DType.BFLOAT16):
        out = model(x)
        assert out.dtype == orca.DType.BFLOAT16

def test_no_grad_behavior():
    # Verify defaults
    assert orca.is_grad_enabled()
    
    a = orca.Tensor.randn([2, 2], requires_grad=True)
    b = orca.Tensor.randn([2, 2], requires_grad=True)
    c = a + b
    assert c.requires_grad
    
    # Test context manager
    with orca.no_grad():
        assert not orca.is_grad_enabled()
        d = a + b
        assert not d.requires_grad
        
    assert orca.is_grad_enabled()
    
    # Test decorator
    @orca.no_grad()
    def compute_no_grad(x, y):
        assert not orca.is_grad_enabled()
        return x + y
        
    e = compute_no_grad(a, b)
    assert not e.requires_grad
    assert orca.is_grad_enabled()

def test_has_nan_or_inf_cpu_f16():
    # Create float16 tensor with nan/inf and test
    a = orca.Tensor.from_list([1.0, float('nan'), 2.0], shape=[3], requires_grad=False)
    a_f16 = a.to_dtype(orca.DType.FLOAT16)
    assert a_f16.dtype == orca.DType.FLOAT16
    assert a_f16.has_nan_or_inf()

    b = orca.Tensor.from_list([1.0, 2.0, 3.0], shape=[3], requires_grad=False)
    b_f16 = b.to_dtype(orca.DType.FLOAT16)
    assert not b_f16.has_nan_or_inf()

def test_tensor_str_formatting():
    t = orca.Tensor.from_list([1.1, 2.2, 3.3, 4.4], shape=[2, 2], requires_grad=False)
    s = str(t)
    assert "tensor([[" in s
    assert "1.1000" in s
    assert "device='cpu'" in s
    assert "dtype=float32" in s
