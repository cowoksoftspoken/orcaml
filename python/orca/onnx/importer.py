import onnx
import orca
import orca.nn as nn
from typing import Dict, List, Any, Union
import numpy as np

class ONNXInterpreter(nn.Module):
    """
    Dynamic execution engine that runs an ONNX computational graph node-by-node.
    Reconstructs high-level child modules (e.g. nn.Linear, nn.Conv2d) where possible
    and falls back to primitive Orca operations for other nodes.
    """
    def __init__(
        self,
        nodes: List[Dict[str, Any]],
        initializers: Dict[str, orca.Tensor],
        child_modules: Dict[str, nn.Module],
        graph_input_name: str,
        graph_output_name: str
    ):
        super().__init__()
        self.nodes = nodes
        self.initializers = initializers
        self.graph_input_name = graph_input_name
        self.graph_output_name = graph_output_name
        
        # Register child modules so parameters are visible to optimizers
        for name, module in child_modules.items():
            setattr(self, name, module)

    def forward(self, x: orca.Tensor) -> orca.Tensor:
        values = {}
        values[self.graph_input_name] = x
        
        # Add initializers to values dict
        for name, tensor in self.initializers.items():
            values[name] = tensor
            
        for node in self.nodes:
            op_type = node['op_type']
            inputs = [
                values[inp] if inp in values else self.initializers.get(inp) 
                for inp in node['inputs']
            ]
            outputs = node['outputs']
            attrs = node['attrs']
            
            # Execute operations
            if op_type == 'Linear':
                linear = getattr(self, node['module_name'])
                res = linear(inputs[0])
            elif op_type == 'Conv2d':
                conv = getattr(self, node['module_name'])
                res = conv(inputs[0])
            elif op_type == 'Add':
                res = inputs[0] + inputs[1]
            elif op_type == 'Sub':
                res = inputs[0] - inputs[1]
            elif op_type == 'Mul':
                res = inputs[0] * inputs[1]
            elif op_type == 'Div':
                res = inputs[0] / inputs[1]
            elif op_type == 'MatMul':
                res = inputs[0] @ inputs[1]
            elif op_type == 'Relu':
                res = inputs[0].relu()
            elif op_type == 'Sigmoid':
                res = inputs[0].sigmoid()
            elif op_type == 'Exp':
                res = inputs[0].exp()
            elif op_type == 'Log':
                res = inputs[0].log()
            elif op_type == 'Neg':
                res = -inputs[0]
            elif op_type == 'Reshape':
                # Target shape could be in inputs[1]
                shape = inputs[1]
                if isinstance(shape, orca.Tensor):
                    shape_list = [int(v) for v in shape.to_list()]
                else:
                    shape_list = [int(v) for v in shape]
                res = inputs[0].reshape(shape_list)
            elif op_type == 'Transpose':
                perm = attrs.get('perm', [1, 0])
                res = inputs[0].transpose(perm[0], perm[1])
            elif op_type == 'Expand':
                shape = inputs[1]
                if isinstance(shape, orca.Tensor):
                    shape_list = [int(v) for v in shape.to_list()]
                else:
                    shape_list = [int(v) for v in shape]
                res = inputs[0].expand(shape_list)
            elif op_type == 'ReduceSum':
                res = inputs[0].sum_to_shape([1])
            else:
                node_name = node.get("module_name") or (outputs[0] if outputs else "<unknown>")
                raise NotImplementedError(
                    f"Unsupported ONNX op '{op_type}' in imported graph node '{node_name}'"
                )

            values[outputs[0]] = res
            
        return values[self.graph_output_name]

def import_onnx(filepath: str) -> nn.Module:
    """
    Imports an ONNX model file and reconstructs it as an executable, autograd-safe Orca Module.
    """
    model_proto = onnx.load(filepath)
    graph = model_proto.graph
    
    # 1. Parse Initializers
    initializers = {}
    for init in graph.initializer:
        from onnx.numpy_helper import to_array
        arr = to_array(init)
        shape = list(init.dims)
        # Handle scalar (empty shape)
        if not shape:
            shape = [1]
        tensor = orca.Tensor.from_list(arr.flatten().tolist(), shape=shape)
        initializers[init.name] = tensor

    # 2. Identify Graph Inputs/Outputs
    graph_input_name = graph.input[0].name
    graph_output_name = graph.output[0].name

    # 3. Perform Pattern Matching & Graph Reconstruction
    nodes = list(graph.node)
    interpreted_nodes = []
    child_modules = {}
    module_counter = 0
    skipped_nodes = set()
    
    for idx, node in enumerate(nodes):
        if idx in skipped_nodes:
            continue
            
        op_type = node.op_type
        inputs = list(node.input)
        outputs = list(node.output)
        
        # Parse attributes
        attrs = {}
        for attr in node.attribute:
            from onnx import helper
            attrs[attr.name] = helper.get_attribute_value(attr)
            
        # Pattern 1: Gemm Node (Linear with weights and optional bias)
        if op_type == 'Gemm':
            w_name = inputs[1]
            if w_name in initializers:
                w_tensor = initializers[w_name]
                
                # Gemm standard attributes
                transB = attrs.get('transB', 0)
                if transB == 1:
                    w_tensor = w_tensor.transpose(0, 1)
                    
                in_features = w_tensor.shape[0]
                out_features = w_tensor.shape[1]
                
                has_bias = len(inputs) > 2
                b_tensor = None
                if has_bias:
                    b_name = inputs[2]
                    if b_name in initializers:
                        b_tensor = initializers[b_name]
                        
                linear = nn.Linear(in_features, out_features, bias=has_bias)
                linear.weight.tensor = w_tensor
                if has_bias and b_tensor is not None:
                    linear.bias.tensor = b_tensor.reshape([1, out_features])
                    
                mod_name = f"linear_{module_counter}"
                module_counter += 1
                child_modules[mod_name] = linear
                
                interpreted_nodes.append({
                    'op_type': 'Linear',
                    'inputs': [inputs[0]],
                    'outputs': outputs,
                    'module_name': mod_name,
                    'attrs': attrs
                })
                continue

        # Pattern 2: MatMul followed by (optional Expand and) Add
        if op_type == 'MatMul':
            w_name = inputs[1]
            if w_name in initializers:
                w_tensor = initializers[w_name]
                in_features, out_features = w_tensor.shape
                
                matched_add = False
                for next_idx in range(idx + 1, len(nodes)):
                    next_node = nodes[next_idx]
                    if next_node.op_type == 'Add' and outputs[0] in next_node.input:
                        # Find the other input of the Add node
                        matmul_idx = list(next_node.input).index(outputs[0])
                        other_idx = 1 - matmul_idx
                        other_in = list(next_node.input)[other_idx]
                        
                        b_tensor = None
                        b_name = None
                        
                        # Case 1: The other input is directly the bias initializer
                        if other_in in initializers:
                            b_name = other_in
                            b_tensor = initializers[b_name]
                        else:
                            # Case 2: The other input comes from an Expand node
                            for prev_idx in range(idx + 1, next_idx):
                                p_node = nodes[prev_idx]
                                if p_node.op_type == 'Expand' and list(p_node.output)[0] == other_in:
                                    exp_in = list(p_node.input)[0]
                                    if exp_in in initializers:
                                        b_name = exp_in
                                        b_tensor = initializers[b_name]
                                        skipped_nodes.add(prev_idx)
                                        break
                                        
                        if b_tensor is not None:
                            linear = nn.Linear(in_features, out_features, bias=True)
                            linear.weight.tensor = w_tensor
                            linear.bias.tensor = b_tensor.reshape([1, out_features])
                            
                            mod_name = f"linear_{module_counter}"
                            module_counter += 1
                            child_modules[mod_name] = linear
                            
                            interpreted_nodes.append({
                                'op_type': 'Linear',
                                'inputs': [inputs[0]],
                                'outputs': list(next_node.output),
                                'module_name': mod_name,
                                'attrs': attrs
                            })
                            skipped_nodes.add(next_idx)
                            matched_add = True
                            break
                            
                if matched_add:
                    continue
                else:
                    # MatMul only (Linear without bias)
                    linear = nn.Linear(in_features, out_features, bias=False)
                    linear.weight.tensor = w_tensor
                    
                    mod_name = f"linear_{module_counter}"
                    module_counter += 1
                    child_modules[mod_name] = linear
                    
                    interpreted_nodes.append({
                        'op_type': 'Linear',
                        'inputs': [inputs[0]],
                        'outputs': outputs,
                        'module_name': mod_name,
                        'attrs': attrs
                    })
                    continue

        # Pattern 3: Conv (mapped to Conv2d)
        if op_type == 'Conv':
            w_name = inputs[1]
            if w_name in initializers:
                w_tensor = initializers[w_name]
                out_channels, in_channels, kh, kw = w_tensor.shape
                
                strides = attrs.get('strides', [1, 1])
                pads = attrs.get('pads', [0, 0, 0, 0])
                dilations = attrs.get('dilations', [1, 1])
                padding = pads[0] if len(pads) > 0 else 0
                
                has_bias = len(inputs) > 2
                b_tensor = None
                if has_bias:
                    b_name = inputs[2]
                    if b_name in initializers:
                        b_tensor = initializers[b_name]
                        
                conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kh,
                    stride=strides[0],
                    padding=padding,
                    dilation=dilations[0],
                    bias=has_bias
                )
                conv.weight.tensor = w_tensor
                if has_bias and b_tensor is not None:
                    conv.bias.tensor = b_tensor.reshape([out_channels, 1, 1])
                    
                mod_name = f"conv_{module_counter}"
                module_counter += 1
                child_modules[mod_name] = conv
                
                interpreted_nodes.append({
                    'op_type': 'Conv2d',
                    'inputs': [inputs[0]],
                    'outputs': outputs,
                    'module_name': mod_name,
                    'attrs': attrs
                })
                continue

        # Fallback to primitive nodes
        interpreted_nodes.append({
            'op_type': op_type,
            'inputs': inputs,
            'outputs': outputs,
            'attrs': attrs
        })

    return ONNXInterpreter(interpreted_nodes, initializers, child_modules, graph_input_name, graph_output_name)
