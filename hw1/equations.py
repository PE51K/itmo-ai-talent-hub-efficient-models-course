from functools import reduce

import numpy as np


def flops_per_conv(S: int | np.ndarray, B: int | np.ndarray, pad: int, stride: int, k: int, in_ch: int, out_ch: int) -> int | np.ndarray:
    """
    Calc flops per Conv2d layer

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    return (
        ((S + pad - k // 2) // stride) ** 2 # number of kernel applications per 1 kernel in 1 item
    ) * (
        2 * (in_ch * k ** 2) # number of flops per kernel (in_ch * k^2 MACs, 1 MAC = 2 flops)
    ) * out_ch * B # repeat per each item in batch and per each kernel


def flops_per_max_pool(S: int | np.ndarray, B: int | np.ndarray, pad: int, stride: int, k: int, in_ch: int) -> int | np.ndarray:
    """
    Calc flops per MaxPool2d layer

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input channels
    """
    return (
        ((S + pad - k // 2) // stride) ** 2 # number of pool applications per 1 channel in 1 item
    ) * (
        (k ** 2) - 1 # number of flops per one pool (k^2 - 1 comparisons)
    ) * in_ch * B # repeat per each item in batch and per each channel


def flops_per_relu(S: int | np.ndarray, B: int | np.ndarray, in_ch: int) -> int | np.ndarray:
    """
    Calc flops per ReLU layer

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        in_ch: num of input channels
    """
    return (
        in_ch * S ** 2 # number of values in one item (1 comparison with 0 per value)
    ) * B # repeat per each item in batch


def flops_per_avg_pool(S: int | np.ndarray, B: int | np.ndarray, in_ch: int) -> int | np.ndarray:
    """
    Calc flops per AvgPool2d layer

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        in_ch: num of input channels
    """
    return (
        S ** 2 # number of flops in one channel (sum all (s^2 - 1) and divide (1))
    ) * in_ch * B # repeat per each channel in each item in batch


def flops_per_linear(B: int | np.ndarray, in_ch: int, out_ch: int) -> int | np.ndarray:
    """
    Calc flops per Linear layer (with bias)

    Args:
        B: batch size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    return (
        2 * in_ch # number of flops per one scalar product ((2 * in_ch - 1) + 1 flop per bias)
    ) * out_ch * B # repeat per item in batch


def flops(image_size: int | np.ndarray, batch: int | np.ndarray) -> int | np.ndarray:
    """
    Calc flops per one forward pass of model from models.py

    Args:
        image_size: input shape (w & h, only w=h supported)
        batch: batch size
    """
    S, B = image_size, batch
    return (
        flops_per_conv(S, B, pad=3, stride=2, k=7, in_ch=3, out_ch=32) # -> S/2
        + flops_per_relu(S // 2, B, in_ch=32)
        + flops_per_max_pool(S // 2, B, pad=1, stride=2, k=3, in_ch=32) # -> S/4
        + flops_per_conv(S // 4, B, pad=2, stride=1, k=5, in_ch=32, out_ch=64)
        + flops_per_relu(S // 4, B, in_ch=64)
        + flops_per_conv(S // 4, B, pad=1, stride=2, k=3, in_ch=64, out_ch=128) # -> S/8
        + flops_per_relu(S // 8, B, in_ch=128)
        + flops_per_conv(S // 8, B, pad=0, stride=1, k=1, in_ch=128, out_ch=256)
        + flops_per_relu(S // 8, B, in_ch=256)
        + flops_per_conv(S // 8, B, pad=1, stride=2, k=3, in_ch=256, out_ch=256) # -> S/16
        + flops_per_relu(S // 16, B, in_ch=256)
        + flops_per_conv(S // 16, B, pad=0, stride=1, k=1, in_ch=256, out_ch=512)
        + flops_per_relu(S // 16, B, in_ch=512)
        + flops_per_avg_pool(S // 16, B, in_ch=512) # -> 1
        + flops_per_linear(B, in_ch=512, out_ch=256)
        + flops_per_relu(1, B, in_ch=256)
        + flops_per_linear(B, in_ch=256, out_ch=100)
    )


def params_memory_per_conv(k: int, in_ch: int, out_ch: int) -> int:
    """
    Calc memory (bytes) of Conv2d weights (without bias)

    Args:
        k: kernel size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    return (
        in_ch * k ** 2 # weights per one kernel
    ) * out_ch * 4 # repeat per each kernel, 4 bytes per fp32


def params_memory_per_linear(in_ch: int, out_ch: int) -> int:
    """
    Calc memory (bytes) of Linear weights (with bias)

    Args:
        in_ch: num of input channels
        out_ch: num of output channels
    """
    return (
        in_ch * out_ch + out_ch # weights + bias
    ) * 4 # 4 bytes per fp32


def memory_per_conv_forward(S: int | np.ndarray, B: int | np.ndarray, pad: int, stride: int, k: int, in_ch: int, out_ch: int) -> int | np.ndarray:
    """
    Calc memory (bytes) allocated by tensors during Conv2d forward (input + output)

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    return (
        in_ch * S ** 2 # input values in one item
        + out_ch * ((S + 2 * pad - k) // stride + 1) ** 2 # output values in one item
    ) * B * 4 # repeat per each item in batch, 4 bytes per fp32


def memory_per_max_pool_forward(S: int | np.ndarray, B: int | np.ndarray, pad: int, stride: int, k: int, in_ch: int) -> int | np.ndarray:
    """
    Calc memory (bytes) allocated by tensors during MaxPool2d forward (input + output + indices)

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input channels
    """
    # On CUDA max_pool2d always runs max_pool2d_with_indices, so int64 indices are allocated even in inference
    out = ((S + 2 * pad - k) // stride + 1) ** 2 # output values in one channel
    return (
        S ** 2 * 4 # input values in one channel, 4 bytes per fp32
        + out * 4 # output values, 4 bytes per fp32
        + out * 8 # indices, 8 bytes per int64
    ) * in_ch * B # repeat per each channel in each item in batch


def memory_per_relu_forward(S: int | np.ndarray, B: int | np.ndarray, in_ch: int) -> int | np.ndarray:
    """
    Calc memory (bytes) allocated by tensors during ReLU forward (inplace, so input only)

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        in_ch: num of input channels
    """
    return (
        in_ch * S ** 2 # values in one item, output shares input storage
    ) * B * 4 # repeat per each item in batch, 4 bytes per fp32


def memory_per_avg_pool_forward(S: int | np.ndarray, B: int | np.ndarray, in_ch: int) -> int | np.ndarray:
    """
    Calc memory (bytes) allocated by tensors during AvgPool2d forward (input + output)

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        in_ch: num of input channels
    """
    return (
        S ** 2 + 1 # input values + 1 output value in one channel (global pool)
    ) * in_ch * B * 4 # repeat per each channel in each item in batch, 4 bytes per fp32


def memory_per_linear_forward(B: int | np.ndarray, in_ch: int, out_ch: int) -> int | np.ndarray:
    """
    Calc memory (bytes) allocated by tensors during Linear forward (input + output)

    Args:
        B: batch size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    return (
        in_ch + out_ch # input + output values in one item
    ) * B * 4 # repeat per each item in batch, 4 bytes per fp32


def memory(image_size: int | np.ndarray, batch: int | np.ndarray) -> int | np.ndarray:
    """
    Calc max allocated memory per one forward pass of model from models.py

    Args:
        image_size: input shape (w & h, only w=h supported)
        batch: batch size
    """
    S, B = image_size, batch
    params = (
        params_memory_per_conv(k=7, in_ch=3, out_ch=32)
        + params_memory_per_conv(k=5, in_ch=32, out_ch=64)
        + params_memory_per_conv(k=3, in_ch=64, out_ch=128)
        + params_memory_per_conv(k=1, in_ch=128, out_ch=256)
        + params_memory_per_conv(k=3, in_ch=256, out_ch=256)
        + params_memory_per_conv(k=1, in_ch=256, out_ch=512)
        + params_memory_per_linear(in_ch=512, out_ch=256)
        + params_memory_per_linear(in_ch=256, out_ch=100)
    )
    # Caller keeps a reference to the input batch, so its storage isn't freed after the 1st layer
    x = (
        3 * S ** 2 # input values in one item
    ) * B * 4 # repeat per each item in batch, 4 bytes per fp32
    # cuBLAS workspace for Linear, taken from caching allocator on 1st matmul and kept (4096 KiB * 2 + 16 KiB * 8 below sm_90)
    cublas_workspace = 4096 * 1024 * 2 + 16 * 1024 * 8
    # np.maximum instead of max() to keep NumPy broadcasting
    return params + cublas_workspace + np.maximum(
        memory_per_conv_forward(S, B, pad=3, stride=2, k=7, in_ch=3, out_ch=32), # input tensor is x
        x + reduce(np.maximum, [
            memory_per_relu_forward(S // 2, B, in_ch=32),
            memory_per_max_pool_forward(S // 2, B, pad=1, stride=2, k=3, in_ch=32),
            memory_per_conv_forward(S // 4, B, pad=2, stride=1, k=5, in_ch=32, out_ch=64),
            memory_per_relu_forward(S // 4, B, in_ch=64),
            memory_per_conv_forward(S // 4, B, pad=1, stride=2, k=3, in_ch=64, out_ch=128),
            memory_per_relu_forward(S // 8, B, in_ch=128),
            memory_per_conv_forward(S // 8, B, pad=0, stride=1, k=1, in_ch=128, out_ch=256),
            memory_per_relu_forward(S // 8, B, in_ch=256),
            memory_per_conv_forward(S // 8, B, pad=1, stride=2, k=3, in_ch=256, out_ch=256),
            memory_per_relu_forward(S // 16, B, in_ch=256),
            memory_per_conv_forward(S // 16, B, pad=0, stride=1, k=1, in_ch=256, out_ch=512),
            memory_per_relu_forward(S // 16, B, in_ch=512),
            memory_per_avg_pool_forward(S // 16, B, in_ch=512), # Flatten returns a view, no new allocation
            memory_per_linear_forward(B, in_ch=512, out_ch=256),
            memory_per_relu_forward(1, B, in_ch=256),
            memory_per_linear_forward(B, in_ch=256, out_ch=100),
        ]),
    )


def bytes_moved_per_conv(S: int | np.ndarray, B: int | np.ndarray, pad: int, stride: int, k: int, in_ch: int, out_ch: int) -> int | np.ndarray:
    """
    Calc DRAM traffic (bytes) of Conv2d kernel (without bias), assuming each element is read/written once

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    return (
        (
            in_ch * S ** 2 # read input values in one item, padding zeros aren't stored
            + out_ch * ((S + 2 * pad - k) // stride + 1) ** 2 # write output values in one item
        ) * B # repeat per each item in batch
        + (in_ch * k ** 2) * out_ch # read weights once, independent of batch
    ) * 4 # 4 bytes per fp32


def bytes_moved_per_max_pool(S: int | np.ndarray, B: int | np.ndarray, pad: int, stride: int, k: int, in_ch: int) -> int | np.ndarray:
    """
    Calc DRAM traffic (bytes) of MaxPool2d kernel (input + output + indices), assuming each element is read/written once

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input channels
    """
    out = ((S + 2 * pad - k) // stride + 1) ** 2 # output values in one channel
    return (
        S ** 2 * 4 # read input values, overlapping windows hit L2, 4 bytes per fp32
        + out * 4 # write output values, 4 bytes per fp32
        + out * 8 # write indices, max_pool2d_with_indices runs on CUDA, 8 bytes per int64
    ) * in_ch * B # repeat per each channel in each item in batch


def bytes_moved_per_relu(S: int | np.ndarray, B: int | np.ndarray, in_ch: int) -> int | np.ndarray:
    """
    Calc DRAM traffic (bytes) of ReLU kernel (inplace saves allocation, not traffic)

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        in_ch: num of input channels
    """
    return (
        2 * in_ch * S ** 2 # read + write back values in one item
    ) * B * 4 # repeat per each item in batch, 4 bytes per fp32


def bytes_moved_per_avg_pool(S: int | np.ndarray, B: int | np.ndarray, in_ch: int) -> int | np.ndarray:
    """
    Calc DRAM traffic (bytes) of AvgPool2d kernel (global pool)

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        in_ch: num of input channels
    """
    return (
        S ** 2 + 1 # read input values + write 1 output value in one channel
    ) * in_ch * B * 4 # repeat per each channel in each item in batch, 4 bytes per fp32


def bytes_moved_per_linear(B: int | np.ndarray, in_ch: int, out_ch: int) -> int | np.ndarray:
    """
    Calc DRAM traffic (bytes) of Linear kernel (with bias), assuming each element is read/written once

    Args:
        B: batch size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    return (
        (in_ch + out_ch) * B # read input + write output values, repeat per each item in batch
        + in_ch * out_ch + out_ch # read weights + bias once, independent of batch
    ) * 4 # 4 bytes per fp32


def kernels(S: int | np.ndarray, B: int | np.ndarray) -> list[tuple[int | np.ndarray, int | np.ndarray]]:
    """
    Calc (flops, bytes moved) per each CUDA kernel launched in one forward pass of model from models.py

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
    """
    return [
        (
            flops_per_conv(S, B, pad=3, stride=2, k=7, in_ch=3, out_ch=32),
            bytes_moved_per_conv(S, B, pad=3, stride=2, k=7, in_ch=3, out_ch=32),
        ),
        (
            flops_per_relu(S // 2, B, in_ch=32),
            bytes_moved_per_relu(S // 2, B, in_ch=32),
        ),
        (
            flops_per_max_pool(S // 2, B, pad=1, stride=2, k=3, in_ch=32),
            bytes_moved_per_max_pool(S // 2, B, pad=1, stride=2, k=3, in_ch=32),
        ),
        (
            flops_per_conv(S // 4, B, pad=2, stride=1, k=5, in_ch=32, out_ch=64),
            bytes_moved_per_conv(S // 4, B, pad=2, stride=1, k=5, in_ch=32, out_ch=64),
        ),
        (
            flops_per_relu(S // 4, B, in_ch=64),
            bytes_moved_per_relu(S // 4, B, in_ch=64),
        ),
        (
            flops_per_conv(S // 4, B, pad=1, stride=2, k=3, in_ch=64, out_ch=128),
            bytes_moved_per_conv(S // 4, B, pad=1, stride=2, k=3, in_ch=64, out_ch=128),
        ),
        (
            flops_per_relu(S // 8, B, in_ch=128),
            bytes_moved_per_relu(S // 8, B, in_ch=128),
        ),
        (
            flops_per_conv(S // 8, B, pad=0, stride=1, k=1, in_ch=128, out_ch=256),
            bytes_moved_per_conv(S // 8, B, pad=0, stride=1, k=1, in_ch=128, out_ch=256),
        ),
        (
            flops_per_relu(S // 8, B, in_ch=256),
            bytes_moved_per_relu(S // 8, B, in_ch=256),
        ),
        (
            flops_per_conv(S // 8, B, pad=1, stride=2, k=3, in_ch=256, out_ch=256),
            bytes_moved_per_conv(S // 8, B, pad=1, stride=2, k=3, in_ch=256, out_ch=256),
        ),
        (
            flops_per_relu(S // 16, B, in_ch=256),
            bytes_moved_per_relu(S // 16, B, in_ch=256),
        ),
        (
            flops_per_conv(S // 16, B, pad=0, stride=1, k=1, in_ch=256, out_ch=512),
            bytes_moved_per_conv(S // 16, B, pad=0, stride=1, k=1, in_ch=256, out_ch=512),
        ),
        (
            flops_per_relu(S // 16, B, in_ch=512),
            bytes_moved_per_relu(S // 16, B, in_ch=512),
        ),
        (
            flops_per_avg_pool(S // 16, B, in_ch=512),
            bytes_moved_per_avg_pool(S // 16, B, in_ch=512),
        ),
        # Flatten returns a view, no kernel launch
        (
            flops_per_linear(B, in_ch=512, out_ch=256),
            bytes_moved_per_linear(B, in_ch=512, out_ch=256),
        ),
        (
            flops_per_relu(1, B, in_ch=256),
            bytes_moved_per_relu(1, B, in_ch=256),
        ),
        (
            flops_per_linear(B, in_ch=256, out_ch=100),
            bytes_moved_per_linear(B, in_ch=256, out_ch=100),
        ),
    ]


def latency(image_size: int | np.ndarray, batch: int | np.ndarray, theta: dict[str, float]) -> float | np.ndarray:
    """
    Calc latency (seconds) of one forward pass of model from models.py, roofline per each kernel + launch overhead

    Args:
        image_size: input shape (w & h, only w=h supported)
        batch: batch size
        theta: t_launch (seconds per kernel launch), flops_per_s (effective compute throughput), bytes_per_s (effective DRAM bandwidth)
    """
    S, B = image_size, batch
    return sum(
        theta["t_launch"] # fixed overhead per kernel launch, dominates in launch-bound regime
        + np.maximum(
            f / theta["flops_per_s"], # compute-bound time
            m / theta["bytes_per_s"], # memory-bound time
        ) # compute and memory traffic overlap inside kernel, so the slower one wins
        for f, m in kernels(S, B)
    )


def energy(image_size: int | np.ndarray, batch: int | np.ndarray, theta_energy: dict[str, float | dict[str, float]]) -> float | np.ndarray:
    """
    Calc energy (joules) of one forward pass of model from models.py, measured on the whole GPU

    Args:
        image_size: input shape (w & h, only w=h supported)
        batch: batch size
        theta_energy: j_per_flop (dynamic energy per flop), j_per_byte (dynamic energy per DRAM byte),
            p_static (watts drawn by whole GPU regardless of load), theta (latency params, see latency())
    """
    S, B = image_size, batch
    ks = kernels(S, B)
    return (
        theta_energy["j_per_flop"] * sum(f for f, _ in ks) # dynamic energy of compute
        + theta_energy["j_per_byte"] * sum(m for _, m in ks) # dynamic energy of DRAM traffic
        + theta_energy["p_static"] * latency(S, B, theta_energy["theta"]) # static power over pass duration
    )
