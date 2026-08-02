import math
from orca.tensor import Tensor
from .parameter import Parameter

def _init_tensor(tensor_or_param, new_data_fn):
    is_param = isinstance(tensor_or_param, Parameter)
    tensor = tensor_or_param.tensor if is_param else tensor_or_param
    
    new_t = new_data_fn(tensor.shape, tensor.dtype, tensor.device, tensor.requires_grad)
    
    if is_param:
        tensor_or_param.update(new_t)
        return tensor_or_param
    return new_t

def normal_(tensor_or_param, mean=0.0, std=1.0):
    """Fills the input Tensor or Parameter with values drawn from a normal distribution."""
    def new_fn(shape, dtype, device, requires_grad):
        return Tensor.randn(shape, mean=mean, std=std, dtype=dtype, device=device, requires_grad=requires_grad)
    return _init_tensor(tensor_or_param, new_fn)

def uniform_(tensor_or_param, a=0.0, b=1.0):
    """Fills the input Tensor or Parameter with values drawn from a uniform distribution."""
    def new_fn(shape, dtype, device, requires_grad):
        return Tensor.rand_uniform(shape, a, b, dtype=dtype, device=device, requires_grad=requires_grad)
    return _init_tensor(tensor_or_param, new_fn)

def zeros_(tensor_or_param):
    """Fills the input Tensor or Parameter with zeros."""
    def new_fn(shape, dtype, device, requires_grad):
        return Tensor.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    return _init_tensor(tensor_or_param, new_fn)

def ones_(tensor_or_param):
    """Fills the input Tensor or Parameter with ones."""
    def new_fn(shape, dtype, device, requires_grad):
        return Tensor.ones(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    return _init_tensor(tensor_or_param, new_fn)

def kaiming_uniform_(tensor_or_param, a=0.0, mode='fan_in', nonlinearity='leaky_relu'):
    """Fills the input Tensor or Parameter with values according to the He (Kaiming) uniform initialization."""
    is_param = isinstance(tensor_or_param, Parameter)
    tensor = tensor_or_param.tensor if is_param else tensor_or_param
    
    fan = _calculate_correct_fan(tensor.shape, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    
    return uniform_(tensor_or_param, -bound, bound)

def kaiming_normal_(tensor_or_param, a=0.0, mode='fan_in', nonlinearity='leaky_relu'):
    """Fills the input Tensor or Parameter with values according to the He (Kaiming) normal initialization."""
    is_param = isinstance(tensor_or_param, Parameter)
    tensor = tensor_or_param.tensor if is_param else tensor_or_param
    
    fan = _calculate_correct_fan(tensor.shape, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    
    return normal_(tensor_or_param, 0.0, std)

def xavier_uniform_(tensor_or_param, gain=1.0):
    """Fills the input Tensor or Parameter with values according to the Xavier (Glorot) uniform initialization."""
    is_param = isinstance(tensor_or_param, Parameter)
    tensor = tensor_or_param.tensor if is_param else tensor_or_param
    
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor.shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    
    return uniform_(tensor_or_param, -bound, bound)

def xavier_normal_(tensor_or_param, gain=1.0):
    """Fills the input Tensor or Parameter with values according to the Xavier (Glorot) normal initialization."""
    is_param = isinstance(tensor_or_param, Parameter)
    tensor = tensor_or_param.tensor if is_param else tensor_or_param
    
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor.shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    return normal_(tensor_or_param, 0.0, std)

def _calculate_fan_in_and_fan_out(shape):
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can only be calculated for tensors with at least 2 dimensions")
    
    if dimensions == 2:
        # Linear weight layout: [in_features, out_features]
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Conv weight layout: [out_channels, in_channels_per_group, kh, kw]
        receptive_field_size = 1
        for s in shape[2:]:
            receptive_field_size *= s
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
        
    return fan_in, fan_out

def _calculate_correct_fan(shape, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")
    
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    return fan_in if mode == 'fan_in' else fan_out

def _calculate_gain(nonlinearity, a=None):
    nonlinearity = nonlinearity.lower()
    if nonlinearity in ['linear', 'conv1d', 'conv2d', 'conv3d']:
        return 1.0
    elif nonlinearity == 'sigmoid':
        return 1.0
    elif nonlinearity == 'tanh':
        return 5.0 / 3.0
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if a is None:
            a = 0.01
        return math.sqrt(2.0 / (1 + a ** 2))
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")
