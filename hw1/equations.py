from functools import reduce

import numpy as np


def out_size(s: int | np.ndarray, pad: int, stride: int, k: int) -> int | np.ndarray:
    """
    Calc output size of Conv2d / MaxPool2d layer (= s / stride, since pad = k // 2, k is odd and s is divisible by stride)

    Args:
        s: layer input size (w & h, only w=h supported)
        pad: padding size
        stride: stride size
        k: kernel size
    """
    return (s + 2 * pad - k) // stride + 1


def flops_per_conv(s: int | np.ndarray, B: int | np.ndarray, pad: int, stride: int, k: int, in_ch: int, out_ch: int) -> int | np.ndarray:
    """
    Calc flops per Conv2d layer

    Args:
        s: layer input size (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    s_out = out_size(s, pad, stride, k)
    return (
        s_out ** 2 * in_ch # apply 1 kernel over all positions and channels
    ) * (
        2 * k ** 2 # k^2 MACs per application
    ) * (
        out_ch # repeat per kernel
    ) * (
        B # per item
    )


def flops_per_max_pool(s: int | np.ndarray, B: int | np.ndarray, pad: int, stride: int, k: int, in_ch: int) -> int | np.ndarray:
    """
    Calc flops per MaxPool2d layer

    Args:
        s: layer input size (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input channels
    """
    s_out = out_size(s, pad, stride, k)
    return s_out ** 2 * in_ch * (
        k ** 2 - 1 # comparisons per window
    ) * B


def flops_per_avg_pool(s: int | np.ndarray, B: int | np.ndarray, in_ch: int) -> int | np.ndarray:
    """
    Calc flops per AvgPool2d layer (global pool)

    Args:
        s: layer input size (w & h, only w=h supported)
        B: batch size
        in_ch: num of input channels
    """
    return (
        s ** 2 # (s^2 - 1) adds + 1 divide per channel
    ) * in_ch * B


def flops_per_relu(s: int | np.ndarray, B: int | np.ndarray, in_ch: int) -> int | np.ndarray:
    """
    Calc flops per ReLU layer

    Args:
        s: layer input size (w & h, only w=h supported)
        B: batch size
        in_ch: num of input channels
    """
    return (
        s ** 2 # 1 comparison per value
    ) * in_ch * B


def flops_per_linear(B: int | np.ndarray, in_ch: int, out_ch: int) -> int | np.ndarray:
    """
    Calc flops per Linear layer (with bias)

    Args:
        B: batch size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    return (
        2 * in_ch # in_ch mults, (in_ch - 1) adds, 1 bias add
    ) * out_ch * B


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
        in_ch * k ** 2 # weights per kernel
    ) * out_ch * 4


def params_memory_per_linear(in_ch: int, out_ch: int) -> int:
    """
    Calc memory (bytes) of Linear weights (with bias)

    Args:
        in_ch: num of input channels
        out_ch: num of output channels
    """
    return (
        in_ch * out_ch # weights
        + out_ch # bias
    ) * 4


def memory_per_conv_forward(s: int | np.ndarray, B: int | np.ndarray, pad: int, stride: int, k: int, in_ch: int, out_ch: int) -> int | np.ndarray:
    """
    Calc memory (bytes) allocated by tensors during Conv2d forward (input + output)

    Args:
        s: layer input size (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    s_out = out_size(s, pad, stride, k)
    return (
        in_ch * s ** 2 # input
        + out_ch * s_out ** 2 # output
    ) * B * 4


def memory_per_max_pool_forward(s: int | np.ndarray, B: int | np.ndarray, pad: int, stride: int, k: int, in_ch: int) -> int | np.ndarray:
    """
    Calc memory (bytes) allocated by tensors during MaxPool2d forward (input + output + indices)

    Args:
        s: layer input size (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input channels
    """
    # On CUDA max_pool2d always runs max_pool2d_with_indices, so int64 indices are allocated even in inference
    s_out = out_size(s, pad, stride, k)
    return (
        4 * s ** 2 # input, fp32
        + 4 * s_out ** 2 # output, fp32
        + 8 * s_out ** 2 # indices, int64
    ) * in_ch * B


def memory_per_avg_pool_forward(s: int | np.ndarray, B: int | np.ndarray, in_ch: int) -> int | np.ndarray:
    """
    Calc memory (bytes) allocated by tensors during AvgPool2d forward (input + output, global pool)

    Args:
        s: layer input size (w & h, only w=h supported)
        B: batch size
        in_ch: num of input channels
    """
    return (s ** 2 + 1) * in_ch * B * 4


def memory_per_relu_forward(s: int | np.ndarray, B: int | np.ndarray, in_ch: int) -> int | np.ndarray:
    """
    Calc memory (bytes) allocated by tensors during ReLU forward (inplace, so input only)

    Args:
        s: layer input size (w & h, only w=h supported)
        B: batch size
        in_ch: num of input channels
    """
    return (
        s ** 2 * in_ch # output shares input
    ) * B * 4


def memory_per_linear_forward(B: int | np.ndarray, in_ch: int, out_ch: int) -> int | np.ndarray:
    """
    Calc memory (bytes) allocated by tensors during Linear forward (input + output)

    Args:
        B: batch size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    return (in_ch + out_ch) * B * 4


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
    x = 3 * S ** 2 * B * 4
    # cuBLAS workspace for Linear, taken from caching allocator on 1st matmul and kept.
    # Default CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8 from parseChosenWorkspaceSize() in ATen/cuda/CublasHandlePool.cpp, only sm_90 gets 32 MiB
    cublas_workspace = (
        4096 * 1024 * 2 # 2 chunks of 4096 KiB
        + 16 * 1024 * 8 # 8 chunks of 16 KiB
    )
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


def bytes_moved_per_conv(s: int | np.ndarray, B: int | np.ndarray, pad: int, stride: int, k: int, in_ch: int, out_ch: int) -> int | np.ndarray:
    """
    Calc DRAM traffic (bytes) of Conv2d kernel (without bias), assuming each element is read/written once

    Args:
        s: layer input size (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    s_out = out_size(s, pad, stride, k)
    return (
        (
            in_ch * s ** 2 # read input
            + out_ch * s_out ** 2 # write output
        ) * B
        + in_ch * k ** 2 * out_ch # read weights once
    ) * 4


def bytes_moved_per_linear(B: int | np.ndarray, in_ch: int, out_ch: int) -> int | np.ndarray:
    """
    Calc DRAM traffic (bytes) of Linear kernel (with bias), assuming each element is read/written once

    Args:
        B: batch size
        in_ch: num of input channels
        out_ch: num of output channels
    """
    return (
        (in_ch + out_ch) * B
        + in_ch * out_ch + out_ch # read weights + bias once
    ) * 4


def bytes_moved_per_relu(s: int | np.ndarray, B: int | np.ndarray, in_ch: int) -> int | np.ndarray:
    """
    Calc DRAM traffic (bytes) of ReLU kernel (inplace saves allocation, not traffic)

    Args:
        s: layer input size (w & h, only w=h supported)
        B: batch size
        in_ch: num of input channels
    """
    return (
        2 * s ** 2 * in_ch # read + write back
    ) * B * 4


def bytes_moved_per_max_pool(s: int | np.ndarray, B: int | np.ndarray, pad: int, stride: int, k: int, in_ch: int) -> int | np.ndarray:
    """
    Calc DRAM traffic (bytes) of MaxPool2d kernel: read input, write output + indices, so same as allocated tensors

    Args:
        s: layer input size (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input channels
    """
    return memory_per_max_pool_forward(s, B, pad, stride, k, in_ch)


def bytes_moved_per_avg_pool(s: int | np.ndarray, B: int | np.ndarray, in_ch: int) -> int | np.ndarray:
    """
    Calc DRAM traffic (bytes) of AvgPool2d kernel: read input, write output, so same as allocated tensors

    Args:
        s: layer input size (w & h, only w=h supported)
        B: batch size
        in_ch: num of input channels
    """
    return memory_per_avg_pool_forward(s, B, in_ch)


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
