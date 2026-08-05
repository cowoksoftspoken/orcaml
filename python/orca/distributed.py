"""Distributed data-parallel helpers for Orca Python modules."""
import socket
import struct
import time
from typing import Iterable

import orca
from .nn.module import Module


_COUNT_FORMAT = "!Q"
_COUNT_BYTES = struct.calcsize(_COUNT_FORMAT)
_FLOAT_BYTES = struct.calcsize("!f")
_DEFAULT_TIMEOUT_SECONDS = 30.0


def _validate_timeout(timeout: float) -> float:
    if isinstance(timeout, bool):
        raise TypeError("timeout must be a positive number")

    try:
        value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise TypeError("timeout must be a positive number") from exc

    if value <= 0.0:
        raise ValueError("timeout must be positive")
    return value


def _validate_process_group(rank: int, world_size: int) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rank must be an integer")
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError("world_size must be an integer")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size})")


def _parse_master_addr(master_addr: str) -> tuple[str, int]:
    if not isinstance(master_addr, str):
        raise TypeError("master_addr must be a string in 'host:port' format")

    try:
        host, port_text = master_addr.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError("master_addr must use 'host:port' format") from exc

    if not host:
        raise ValueError("master_addr host cannot be empty")

    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("master_addr port must be an integer") from exc

    if port <= 0 or port > 65535:
        raise ValueError("master_addr port must be in the range [1, 65535]")

    return host, port


def _recv_exact(connection: socket.socket, num_bytes: int) -> bytes:
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")

    buffer = bytearray()
    while len(buffer) < num_bytes:
        try:
            packet = connection.recv(num_bytes - len(buffer))
        except socket.timeout as exc:
            raise RuntimeError(
                f"Timed out while reading {num_bytes} bytes from distributed peer"
            ) from exc
        except OSError as exc:
            raise RuntimeError("Failed to read from distributed peer") from exc

        if not packet:
            raise RuntimeError(
                f"Distributed peer closed connection after {len(buffer)} of {num_bytes} bytes"
            )

        buffer.extend(packet)

    return bytes(buffer)


def _pack_floats(values: Iterable[float]) -> bytes:
    values = list(values)
    if not values:
        return b""
    return struct.pack(f"!{len(values)}f", *values)


def _recv_gradient(connection: socket.socket, expected_elements: int) -> list[float]:
    count_bytes = _recv_exact(connection, _COUNT_BYTES)
    received_elements = struct.unpack(_COUNT_FORMAT, count_bytes)[0]
    if received_elements != expected_elements:
        raise RuntimeError(
            "Gradient size mismatch during all_reduce_gradients: "
            f"expected {expected_elements} elements, got {received_elements}"
        )

    payload = _recv_exact(connection, received_elements * _FLOAT_BYTES)
    if received_elements == 0:
        return []
    return list(struct.unpack(f"!{received_elements}f", payload))


def _send_gradient(connection: socket.socket, values: list[float]) -> None:
    try:
        connection.sendall(struct.pack(_COUNT_FORMAT, len(values)))
        connection.sendall(_pack_floats(values))
    except OSError as exc:
        raise RuntimeError("Failed to write gradient frame to distributed peer") from exc


def _tensor_from_gradient(values: list[float], grad):
    tensor = orca.Tensor.from_list(values, shape=grad.shape, device=grad.device)
    if tensor.dtype != grad.dtype:
        tensor = tensor.to_dtype(grad.dtype)
    return tensor


class DistributedDataParallel(Module):
    """DistributedDataParallel wrapper for Orca modules.

    Args:
        module: Module to wrap.
        rank: Rank of the current worker.
        world_size: Total number of workers.
        master_addr: TCP rendezvous address in ``host:port`` format.
        timeout: Socket timeout in seconds.
    """

    def __init__(
        self,
        module: Module,
        rank: int = 0,
        world_size: int = 1,
        master_addr: str = "127.0.0.1:18888",
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ):
        super().__init__()
        if not isinstance(module, Module):
            raise TypeError("module must be an orca.nn.Module instance")
        _validate_process_group(rank, world_size)

        self.module = module
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.timeout = _validate_timeout(timeout)

        self.host, self.port = _parse_master_addr(master_addr)
        self.sockets = []
        self._connected = False

    def _connect(self):
        if self._connected or self.world_size <= 1:
            return

        if self.rank == 0:
            accepted_sockets = []
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    server_socket.settimeout(self.timeout)
                    server_socket.bind((self.host, self.port))
                    server_socket.listen(self.world_size - 1)

                    for _ in range(1, self.world_size):
                        connection, _ = server_socket.accept()
                        connection.settimeout(self.timeout)
                        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        accepted_sockets.append(connection)
            except OSError as exc:
                for connection in accepted_sockets:
                    connection.close()
                raise RuntimeError(
                    f"Rank 0 failed to initialize distributed listener at {self.master_addr}"
                ) from exc

            self.sockets.extend(accepted_sockets)
        else:
            deadline = time.monotonic() + self.timeout
            last_error = None

            while time.monotonic() < deadline:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(min(1.0, self.timeout))
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                try:
                    client_socket.connect((self.host, self.port))
                    client_socket.settimeout(self.timeout)
                    self.sockets.append(client_socket)
                    break
                except OSError as exc:
                    last_error = exc
                    client_socket.close()
                    time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            else:
                raise RuntimeError(
                    f"Rank {self.rank} failed to connect to master at {self.master_addr}"
                ) from last_error

        self._connected = True

    def close(self):
        """Close all distributed sockets owned by this wrapper."""
        for connection in self.sockets:
            connection.close()
        self.sockets.clear()
        self._connected = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def all_reduce_gradients(self):
        """Synchronize and average gradients across all workers."""
        if self.world_size <= 1:
            return

        self._connect()
        inv_scale = 1.0 / self.world_size

        for param in self.parameters():
            grad = param.tensor.grad()
            if grad is None:
                continue

            grad_data = grad.to_list()
            num_elements = len(grad_data)

            if self.rank == 0:
                summed_grad = list(grad_data)

                for connection in self.sockets:
                    received_grad = _recv_gradient(connection, num_elements)
                    for element_index, value in enumerate(received_grad):
                        summed_grad[element_index] += value

                avg_grad = [value * inv_scale for value in summed_grad]

                for connection in self.sockets:
                    _send_gradient(connection, avg_grad)

                param.tensor.set_grad(_tensor_from_gradient(avg_grad, grad))
            else:
                connection = self.sockets[0]
                _send_gradient(connection, grad_data)
                avg_grad = _recv_gradient(connection, num_elements)
                param.tensor.set_grad(_tensor_from_gradient(avg_grad, grad))


__all__ = ["DistributedDataParallel"]
