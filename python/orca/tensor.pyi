from typing import Optional, List, Union

class DType:
    """
    Data type of a Tensor (e.g., Float32, Float16).
    """
    FLOAT32: 'DType'
    FLOAT64: 'DType'
    FLOAT16: 'DType'
    BFLOAT16: 'DType'
    INT32: 'DType'
    INT64: 'DType'
    UINT8: 'DType'
    BOOL: 'DType'

    def __init__(self, name: str): ...
    def __str__(self) -> str: ...

class Device:
    """
    Device on which a Tensor is allocated (e.g., 'cpu', 'gpu').
    """
    def __init__(self, name: str): ...
    def __str__(self) -> str: ...

class Tensor:
    """
    A multi-dimensional array of elements with automatic differentiation support.
    This is the core data structure of Orcalib.
    """

    @staticmethod
    def zeros(shape: List[int], dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> 'Tensor':
        """
        Creates a tensor filled with zeros.
        
        Args:
            shape (List[int]): The shape of the output tensor.
            dtype (Optional[DType]): The desired data type. Defaults to Float32.
            device (Optional[Device]): The device to place the tensor on ('cpu' or 'gpu').
            requires_grad (bool): If True, operations on this tensor will be recorded for autograd.
            
        Returns:
            Tensor: A new tensor filled with zeros.
        """
        ...
    
    @staticmethod
    def ones(shape: List[int], dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> 'Tensor':
        """
        Creates a tensor filled with ones.
        
        Args:
            shape (List[int]): The shape of the output tensor.
            dtype (Optional[DType]): The desired data type. Defaults to Float32.
            device (Optional[Device]): The device to place the tensor on ('cpu' or 'gpu').
            requires_grad (bool): If True, operations on this tensor will be recorded for autograd.
            
        Returns:
            Tensor: A new tensor filled with ones.
        """
        ...

    @staticmethod
    def scalar(value: float, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> 'Tensor':
        """
        Creates a 0-dimensional scalar tensor.
        
        Args:
            value (float): The scalar value.
            dtype (Optional[DType]): The desired data type. Defaults to Float32.
            device (Optional[Device]): The device to place the tensor on ('cpu' or 'gpu').
            requires_grad (bool): If True, operations on this tensor will be recorded for autograd.
            
        Returns:
            Tensor: A scalar tensor.
        """
        ...

    @staticmethod
    def randn(shape: List[int], mean: float = 0.0, std: float = 1.0, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> 'Tensor':
        """
        Creates a tensor with elements drawn from a normal distribution.
        
        Args:
            shape (List[int]): The shape of the output tensor.
            mean (float): Mean of the normal distribution.
            std (float): Standard deviation of the normal distribution.
            dtype (Optional[DType]): The desired data type. Defaults to Float32.
            device (Optional[Device]): The device to place the tensor on ('cpu' or 'gpu').
            requires_grad (bool): If True, operations on this tensor will be recorded for autograd.
            
        Returns:
            Tensor: A tensor with random normally distributed values.
        """
        ...
    
    @staticmethod
    def rand_uniform(shape: List[int], low: float, high: float, dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> 'Tensor':
        """
        Creates a tensor with elements drawn from a uniform distribution [low, high).
        
        Args:
            shape (List[int]): The shape of the output tensor.
            low (float): Lower bound of the uniform distribution.
            high (float): Upper bound of the uniform distribution.
            dtype (Optional[DType]): The desired data type. Defaults to Float32.
            device (Optional[Device]): The device to place the tensor on ('cpu' or 'gpu').
            requires_grad (bool): If True, operations on this tensor will be recorded for autograd.
            
        Returns:
            Tensor: A tensor with random uniformly distributed values.
        """
        ...
    
    @staticmethod
    def rand_dropout_mask(shape: List[int], p: float, dtype: Optional[DType] = None, device: Optional[Device] = None) -> 'Tensor':
        """
        Creates a dropout mask tensor where elements are randomly set to zero with probability `p`
        and scaled by `1 / (1 - p)` otherwise.
        
        Args:
            shape (List[int]): The shape of the output tensor.
            p (float): Probability of dropping an element.
            dtype (Optional[DType]): The desired data type. Defaults to Float32.
            device (Optional[Device]): The device to place the tensor on ('cpu' or 'gpu').
            
        Returns:
            Tensor: A dropout mask tensor.
        """
        ...
    
    @staticmethod
    def from_list(data: List[float], shape: Optional[List[int]] = None, requires_grad: bool = False, device: Optional[Device] = None) -> 'Tensor':
        """
        Creates a tensor from a 1D Python list.
        
        Args:
            data (List[float]): A flat list of floats.
            shape (Optional[List[int]]): The desired shape. If None, created as a 1D tensor.
            requires_grad (bool): If True, operations on this tensor will be recorded for autograd.
            device (Optional[Device]): The device to place the tensor on ('cpu' or 'gpu').
            
        Returns:
            Tensor: A tensor initialized with the provided data.
        """
        ...
    
    def to(self, device_str: str) -> 'Tensor':
        """
        Moves the tensor to the specified device and returns a new tensor.
        
        Args:
            device_str (str): The target device ('cpu' or 'gpu').
            
        Returns:
            Tensor: A new tensor on the target device.
        """
        ...
    
    @property
    def shape(self) -> List[int]:
        """Returns the shape (dimensions) of the tensor."""
        ...
    
    @property
    def dtype(self) -> DType:
        """Returns the data type of the tensor."""
        ...
    
    @property
    def device(self) -> Device:
        """Returns the device where the tensor is stored."""
        ...
    
    def to_list(self) -> List[float]:
        """
        Converts the tensor data to a flat Python list.
        
        Returns:
            List[float]: A 1D list containing the tensor's data.
        """
        ...

    def to(self, device_str: str) -> 'Tensor':
        """
        Moves the tensor to the specified device.
        
        Args:
            device_str (str): The target device name ('cpu' or 'gpu').
            
        Returns:
            Tensor: A new tensor on the target device.
        """
        ...

    def to_dtype(self, dtype: DType) -> 'Tensor':
        """
        Casts the tensor to the target data type.
        
        Args:
            dtype (DType): The target data type.
            
        Returns:
            Tensor: A new tensor with the target data type.
        """
        ...
    
    def __add__(self, other: 'Tensor') -> 'Tensor': ...
    def __sub__(self, other: 'Tensor') -> 'Tensor': ...
    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor': ...
    def __matmul__(self, other: 'Tensor') -> 'Tensor': ...
    def __truediv__(self, other: 'Tensor') -> 'Tensor': ...
    def __neg__(self) -> 'Tensor': ...
    
    def transpose(self) -> 'Tensor':
        """
        Transposes a 2D tensor (matrix).
        
        Returns:
            Tensor: The transposed tensor.
        """
        ...
        
    def relu(self) -> 'Tensor':
        """
        Applies the Rectified Linear Unit (ReLU) function element-wise.
        
        Returns:
            Tensor: Result of `max(0, x)`.
        """
        ...
        
    def detach(self) -> 'Tensor':
        """
        Returns a new Tensor, detached from the current graph.
        The result will never require gradient.
        
        Returns:
            Tensor: A detached tensor.
        """
        ...

    def set_grad(self, grad: 'Tensor') -> None:
        """
        Overwrites the gradient of this tensor on the autograd tape.
        
        Args:
            grad (Tensor): The new gradient tensor. Must have the same shape.
        """
        ...

    def sqrt(self) -> 'Tensor':
        """
        Applies the square root function element-wise.
        
        Returns:
            Tensor: Result of `sqrt(x)`.
        """
        ...
        
    def sigmoid(self) -> 'Tensor':
        """
        Applies the Sigmoid function element-wise.
        
        Returns:
            Tensor: Result of `1 / (1 + exp(-x))`.
        """
        ...
        
    def exp(self) -> 'Tensor':
        """
        Applies the exponential function element-wise.
        
        Returns:
            Tensor: Result of `exp(x)`.
        """
        ...
        
    def log(self) -> 'Tensor':
        """
        Applies the natural logarithm function element-wise.
        
        Returns:
            Tensor: Result of `ln(x)`.
        """
        ...
    
    def sum(self) -> 'Tensor':
        """
        Computes the sum of all elements in the tensor.
        
        Returns:
            Tensor: A scalar tensor containing the sum.
        """
        ...
        
    def mean(self) -> 'Tensor':
        """
        Computes the mean of all elements in the tensor.
        
        Returns:
            Tensor: A scalar tensor containing the mean.
        """
        ...
    
    def conv2d(self, weight: 'Tensor', bias: Optional['Tensor'] = None, padding: int = 0, stride: int = 1, dilation: int = 1, groups: int = 1) -> 'Tensor':
        """
        Applies a 2D convolution over an input image composed of several input planes.
        
        Args:
            weight (Tensor): Convolution filters of shape (out_channels, in_channels/groups, kH, kW).
            bias (Optional[Tensor]): Optional bias tensor of shape (out_channels).
            padding (int): Implicit padding added to both sides of the input.
            stride (int): Stride of the convolution.
            dilation (int): Spacing between kernel elements.
            groups (int): Number of blocked connections from input to output channels.
            
        Returns:
            Tensor: Output tensor of the convolution.
        """
        ...
    
    def expand(self, shape: List[int]) -> 'Tensor':
        """
        Expands the tensor to the given shape by duplicating elements.
        
        Args:
            shape (List[int]): The desired output shape.
            
        Returns:
            Tensor: The expanded tensor.
        """
        ...
        
    def reshape(self, shape: List[int]) -> 'Tensor':
        """
        Returns a tensor with the same data and number of elements, but with the specified shape.
        
        Args:
            shape (List[int]): The new shape.
            
        Returns:
            Tensor: The reshaped tensor.
        """
        ...
        
    def chunk(self, chunks: int, dim: int = -1) -> List['Tensor']:
        """
        Splits the tensor into a specific number of chunks along a given dimension.
        
        Args:
            chunks (int): Number of chunks to split the tensor into.
            dim (int, optional): The dimension along which to split. Default: -1.
            
        Returns:
            List[Tensor]: A list of tensor chunks.
        """
        ...
        
    def sum_to_shape(self, shape: List[int]) -> 'Tensor':
        """
        Reduces the tensor to the specified shape by summing along the dimensions that differ.
        Used primarily for gradient accumulation during broadcasting.
        
        Args:
            shape (List[int]): The target shape to reduce into.
            
        Returns:
            Tensor: The reduced tensor.
        """
        ...
    
    @property
    def requires_grad(self) -> bool:
        """Returns whether this tensor requires gradient computation."""
        ...
        
    def require_grad(self) -> None:
        """Enables gradient tracking for this tensor in place."""
        ...
    
    def zero_grad(self) -> None:
        """Clears the gradient associated with this tensor."""
        ...
        
    def backward(self) -> None:
        """
        Computes the gradient of this tensor w.r.t. graph leaves.
        This tensor must be a scalar.
        """
        ...
        
    def grad(self) -> Optional['Tensor']:
        """
        Returns the accumulated gradient of this tensor, or None if no gradient was computed.
        
        Returns:
            Optional[Tensor]: A tensor containing the gradients.
        """
        ...
    
    def __repr__(self) -> str: ...
