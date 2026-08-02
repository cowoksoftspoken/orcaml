import socket
import struct
import time
from typing import List
from .nn.module import Module
import orca

class DistributedDataParallel(Module):
    """
    DistributedDataParallel wrapper for Orca modules to enable data-parallel training.
    """
    def __init__(self, module: Module, rank: int = 0, world_size: int = 1, master_addr: str = "127.0.0.1:18888"):
        super().__init__()
        self.module = module
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        
        self.host, self.port = self.master_addr.split(":")
        self.port = int(self.port)
        self.sockets = []
        self._connected = False

    def _connect(self):
        if self._connected or self.world_size <= 1:
            return
            
        if self.rank == 0:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(self.world_size - 1)
            
            for _ in range(1, self.world_size):
                conn, _ = server_socket.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sockets.append(conn)
        else:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for _ in range(20):
                try:
                    client_socket.connect((self.host, self.port))
                    self.sockets.append(client_socket)
                    break
                except ConnectionRefusedError:
                    time.sleep(0.1)
            else:
                raise RuntimeError(f"Rank {self.rank} failed to connect to master at {self.master_addr}")
                
        self._connected = True

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def all_reduce_gradients(self):
        """
        Synchronizes and averages gradients across all workers using TCP sockets.
        """
        if self.world_size <= 1:
            return
            
        self._connect()
        inv_scale = 1.0 / self.world_size

        for param in self.parameters():
            grad = param.tensor.grad()
            if grad is not None:
                grad_data = grad.to_list()
                n = len(grad_data)
                
                if self.rank == 0:
                    summed_grad = list(grad_data)
                    for conn in self.sockets:
                        data = conn.recv(struct.calcsize("!I"))
                        num_elements = struct.unpack("!I", data)[0]
                        
                        bytes_to_read = num_elements * struct.calcsize("!f")
                        buffer = bytearray()
                        while len(buffer) < bytes_to_read:
                            packet = conn.recv(bytes_to_read - len(buffer))
                            if not packet:
                                break
                            buffer.extend(packet)
                        
                        recv_floats = struct.unpack(f"!{num_elements}f", buffer)
                        for i in range(num_elements):
                            summed_grad[i] += recv_floats[i]
                            
                    avg_grad = [g * inv_scale for g in summed_grad]
                    
                    send_data = struct.pack(f"!{n}f", *avg_grad)
                    for conn in self.sockets:
                        conn.sendall(send_data)
                        
                    avg_grad_tensor = orca.Tensor.from_list(avg_grad, shape=grad.shape, device=grad.device)
                    param.tensor.set_grad(avg_grad_tensor)
                else:
                    conn = self.sockets[0]
                    conn.sendall(struct.pack("!I", n))
                    
                    send_data = struct.pack(f"!{n}f", *grad_data)
                    conn.sendall(send_data)
                    
                    bytes_to_read = n * struct.calcsize("!f")
                    buffer = bytearray()
                    while len(buffer) < bytes_to_read:
                        packet = conn.recv(bytes_to_read - len(buffer))
                        if not packet:
                            break
                        buffer.extend(packet)
                        
                    avg_grad = struct.unpack(f"!{n}f", buffer)
                    
                    avg_grad_tensor = orca.Tensor.from_list(list(avg_grad), shape=grad.shape, device=grad.device)
                    param.tensor.set_grad(avg_grad_tensor)

__all__ = ["DistributedDataParallel"]
