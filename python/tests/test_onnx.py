import pytest
import orca
import orca.nn as nn
import orca.optim as optim
import os

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2, bias=True)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def test_onnx_roundtrip(tmp_path, capsys):
    """Verify that exporting to ONNX and importing back reproduces correct values and gradients."""
    model = SimpleMLP()
    model.eval()
    
    # Generate input
    x = orca.Tensor.randn([2, 4])
    
    # Run original forward
    y_orig = model(x)
    
    # Export path
    onnx_path = os.path.join(tmp_path, "mlp.onnx")
    
    # 1. Export
    orca.onnx.export_onnx(model, x, onnx_path)
    assert os.path.exists(onnx_path)
    assert capsys.readouterr().out == ""
    
    # 2. Import
    imported_model = orca.onnx.import_onnx(onnx_path)
    assert isinstance(imported_model, nn.Module)
    
    # 3. Check parameters
    # The imported model parameters should contain the same values
    orig_params = list(model.parameters())
    imported_params = list(imported_model.parameters())
    print("ORIG PARAMS:")
    for p in orig_params:
        print("  shape:", p.tensor.shape)
    print("IMPORTED PARAMS:")
    for p in imported_params:
        print("  shape:", p.tensor.shape)
    assert len(orig_params) == len(imported_params)
    
    # Compare weights
    w1_orig = orig_params[0].tensor.to_list()
    w1_imp = imported_params[0].tensor.to_list()
    assert len(w1_orig) == len(w1_imp)
    for a, b in zip(w1_orig, w1_imp):
        assert a == pytest.approx(b, abs=1e-5)
        
    # 4. Compare forward outputs
    y_imp = imported_model(x)
    assert y_orig.shape == y_imp.shape
    
    y_orig_list = y_orig.to_list()
    y_imp_list = y_imp.to_list()
    for a, b in zip(y_orig_list, y_imp_list):
        assert a == pytest.approx(b, abs=1e-5)
        
    # 5. Check backward pass capability (Autograd safe)
    # Enable gradient tracking on parameters
    for p in imported_model.parameters():
        p.tensor.require_grad()
        
    # Re-run forward pass with gradient tracking enabled
    y_imp = imported_model(x)
    y_imp_sum = y_imp.sum_to_shape([1])
    y_imp_sum.backward()
    
    # Gradients should have been populated
    for p in imported_model.parameters():
        assert p.tensor.grad() is not None
        assert any(val != 0.0 for val in p.tensor.grad().to_list())


def test_onnx_importer_rejects_unsupported_ops():
    """Unsupported ONNX ops must fail explicitly instead of silently returning identity."""
    from orca.onnx.importer import ONNXInterpreter

    model = ONNXInterpreter(
        nodes=[
            {
                "op_type": "UnsupportedOp",
                "inputs": ["input_0"],
                "outputs": ["output_0"],
                "attrs": {},
            }
        ],
        initializers={},
        child_modules={},
        graph_input_name="input_0",
        graph_output_name="output_0",
    )

    with pytest.raises(NotImplementedError, match="Unsupported ONNX op 'UnsupportedOp'"):
        model(orca.Tensor.ones([1]))
