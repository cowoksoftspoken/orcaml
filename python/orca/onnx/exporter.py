import orca
import orca.nn as nn
from typing import Any, Dict, List, Tuple
import onnx
from onnx import helper, TensorProto
import os

class ModelTracer:
    def __init__(self):
        self.tracing = False
        self.nodes = []
        self.tensor_to_name = {}
        self.initializers = {}  # name -> (val_list, shape, type)
        self.const_counter = 0
        self.node_counter = 0

    def get_tensor_name(self, tensor: orca.Tensor) -> str:
        tid = id(tensor)
        if tid in self.tensor_to_name:
            return self.tensor_to_name[tid]
        
        name = f"node_{self.node_counter}"
        self.node_counter += 1
        self.tensor_to_name[tid] = name
        return name

    def add_node(self, op_type: str, inputs: List[Any], outputs: List[orca.Tensor], attrs: Dict[str, Any] = None):
        if not self.tracing:
            return
            
        inputs_resolved = []
        for inp in inputs:
            if isinstance(inp, orca.Tensor):
                inputs_resolved.append(self.get_tensor_name(inp))
            elif isinstance(inp, (int, float)):
                # Create a constant initializer
                const_name = f"const_{self.const_counter}"
                self.const_counter += 1
                self.initializers[const_name] = ([float(inp)], [], TensorProto.FLOAT)
                inputs_resolved.append(const_name)
            elif isinstance(inp, list) and all(isinstance(v, (int, float)) for v in inp):
                const_name = f"const_{self.const_counter}"
                self.const_counter += 1
                self.initializers[const_name] = ([float(v) for v in inp], [len(inp)], TensorProto.FLOAT)
                inputs_resolved.append(const_name)
            else:
                inputs_resolved.append(str(inp))
                
        outputs_resolved = [self.get_tensor_name(out) for out in outputs]
        
        self.nodes.append({
            'op_type': op_type,
            'inputs': inputs_resolved,
            'outputs': outputs_resolved,
            'attrs': attrs or {}
        })

tracer = ModelTracer()

def get_parameter_names(model: nn.Module, prefix: str = "") -> Dict[int, str]:
    names = {}
    for name, param in model._parameters.items():
        if param is not None:
            names[id(param.tensor)] = f"{prefix}{name}"
    for name, submodule in model._modules.items():
        names.update(get_parameter_names(submodule, prefix=f"{prefix}{submodule.__class__.__name__}_{name}."))
    return names

def patch_tensor_methods():
    original_methods = {}
    
    # 1. Binary methods
    def wrap_binary(name, op_type):
        orig = getattr(orca.Tensor, name)
        original_methods[name] = orig
        def wrapper(self, other):
            res = orig(self, other)
            tracer.add_node(op_type, [self, other], [res])
            return res
        setattr(orca.Tensor, name, wrapper)

    # 2. Reverse binary methods
    def wrap_reverse_binary(name, op_type):
        orig = getattr(orca.Tensor, name)
        original_methods[name] = orig
        def wrapper(self, other):
            res = orig(self, other)
            tracer.add_node(op_type, [other, self], [res])
            return res
        setattr(orca.Tensor, name, wrapper)

    # 3. Unary methods
    def wrap_unary(name, op_type):
        orig = getattr(orca.Tensor, name)
        original_methods[name] = orig
        def wrapper(self):
            res = orig(self)
            tracer.add_node(op_type, [self], [res])
            return res
        setattr(orca.Tensor, name, wrapper)

    # Apply patching
    wrap_binary('__add__', 'Add')
    wrap_reverse_binary('__radd__', 'Add')
    wrap_binary('__sub__', 'Sub')
    wrap_reverse_binary('__rsub__', 'Sub')
    wrap_binary('__mul__', 'Mul')
    wrap_reverse_binary('__rmul__', 'Mul')
    wrap_binary('__matmul__', 'MatMul')
    wrap_binary('__truediv__', 'Div')
    wrap_reverse_binary('__rtruediv__', 'Div')
    
    wrap_unary('relu', 'Relu')
    wrap_unary('sigmoid', 'Sigmoid')
    wrap_unary('exp', 'Exp')
    wrap_unary('log', 'Log')
    wrap_unary('__neg__', 'Neg')

    # Custom methods with shapes/dimensions
    orig_reshape = orca.Tensor.reshape
    original_methods['reshape'] = orig_reshape
    def wrapper_reshape(self, shape):
        res = orig_reshape(self, shape)
        tracer.add_node('Reshape', [self, shape], [res])
        return res
    setattr(orca.Tensor, 'reshape', wrapper_reshape)

    orig_transpose = orca.Tensor.transpose
    original_methods['transpose'] = orig_transpose
    def wrapper_transpose(self, dim0, dim1):
        res = orig_transpose(self, dim0, dim1)
        tracer.add_node('Transpose', [self], [res], attrs={'perm': [dim0, dim1]})
        return res
    setattr(orca.Tensor, 'transpose', wrapper_transpose)

    orig_expand = orca.Tensor.expand
    original_methods['expand'] = orig_expand
    def wrapper_expand(self, shape):
        res = orig_expand(self, shape)
        tracer.add_node('Expand', [self, shape], [res])
        return res
    setattr(orca.Tensor, 'expand', wrapper_expand)

    orig_sum_to_shape = orca.Tensor.sum_to_shape
    original_methods['sum_to_shape'] = orig_sum_to_shape
    def wrapper_sum_to_shape(self, shape):
        res = orig_sum_to_shape(self, shape)
        tracer.add_node('ReduceSum', [self], [res], attrs={'axes': list(range(len(self.shape))), 'keepdims': 1})
        return res
    setattr(orca.Tensor, 'sum_to_shape', wrapper_sum_to_shape)

    return original_methods

def restore_tensor_methods(original_methods):
    for name, orig in original_methods.items():
        setattr(orca.Tensor, name, orig)

def export_onnx(model: nn.Module, dummy_input: orca.Tensor, filepath: str):
    """
    Exports an Orca model to ONNX format (opset 17+) using Tape Tracing.
    """
    model.eval()
    
    # 1. Reset tracer state
    tracer.tracing = True
    tracer.nodes = []
    tracer.tensor_to_name = {}
    tracer.initializers = {}
    tracer.const_counter = 0
    tracer.node_counter = 0

    # 2. Register parameters
    param_names = get_parameter_names(model)
    for tid, name in param_names.items():
        tracer.tensor_to_name[tid] = name
        
    # Get parameter references
    params_by_id = {}
    for param in model.parameters():
        params_by_id[id(param.tensor)] = param.tensor

    # 3. Register inputs/outputs
    input_name = "input_0"
    tracer.tensor_to_name[id(dummy_input)] = input_name

    # 4. Patch methods and trace forward pass
    orig_methods = patch_tensor_methods()
    try:
        output = model(dummy_input)
    finally:
        restore_tensor_methods(orig_methods)
        tracer.tracing = False

    output_name = tracer.get_tensor_name(output)

    # 5. Populate initializers for parameters
    for tid, name in param_names.items():
        tensor = params_by_id[tid]
        tracer.initializers[name] = (tensor.to_list(), tensor.shape, TensorProto.FLOAT)

    # 6. Construct ONNX graph components
    onnx_nodes = []
    for node in tracer.nodes:
        # Create attributes list
        onnx_attrs = []
        for k, v in node['attrs'].items():
            if isinstance(v, list):
                if all(isinstance(x, int) for x in v):
                    onnx_attrs.append(helper.make_attribute(k, v))
                else:
                    onnx_attrs.append(helper.make_attribute(k, [float(x) for x in v]))
            elif isinstance(v, int):
                onnx_attrs.append(helper.make_attribute(k, v))
            elif isinstance(v, float):
                onnx_attrs.append(helper.make_attribute(k, v))
            else:
                onnx_attrs.append(helper.make_attribute(k, str(v)))

        clean_inputs = []
        for inp in node['inputs']:
            clean_inputs.append(inp)

        n = helper.make_node(
            node['op_type'],
            inputs=clean_inputs,
            outputs=node['outputs'],
            name=f"{node['op_type']}_{tracer.node_counter}",
        )
        for attr in onnx_attrs:
            n.attribute.extend([attr])
        onnx_nodes.append(n)

    # Initializers
    onnx_initializers = []
    for name, (vals, shape, dtype) in tracer.initializers.items():
        init = helper.make_tensor(
            name=name,
            data_type=dtype,
            dims=shape,
            vals=vals
        )
        onnx_initializers.append(init)

    # Graph Inputs
    graph_inputs = [
        helper.make_tensor_value_info(input_name, TensorProto.FLOAT, dummy_input.shape)
    ]

    # Graph Outputs
    graph_outputs = [
        helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output.shape)
    ]

    # Assemble graph
    graph_proto = helper.make_graph(
        nodes=onnx_nodes,
        name="orca_model",
        inputs=graph_inputs,
        outputs=graph_outputs,
        initializer=onnx_initializers
    )

    # Assemble model with opset 17
    op = onnx.OperatorSetIdProto()
    op.version = 17
    model_proto = helper.make_model(graph_proto, producer_name="orca", opset_imports=[op])

    # Save to file
    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    onnx.save(model_proto, filepath)
