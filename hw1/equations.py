import torch
import torch.nn as nn


def flops_per_conv(S: int, B: int, pad: int, stride: int, k: int, in_ch: int, out_ch: int) -> int:
    """
    Calc flops per Conv2d layer

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input chanels
        out_ch: num of output chanels
    """
    if (S + pad - k // 2) % stride != 0:
        raise ValueError("(S + pad - k // 2) should be devidable by stride size")
    return ( 
        ((S + pad - k // 2) // stride) ** 2 # number of kernel applications per 1 kernel in 1 item
    ) * (
        2 * (in_ch * k ** 2) - 1 # number of flops per kernel (k scalar products, k-1 scalar sums)
    ) * out_ch * B # repeat per each item in batch and per each kernel


def fllops_per_max_pool(S: int, B: int, pad: int, stride: int, k: int, in_ch: int) -> int:
    """
    Calc flops per MaxPool2d layer

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        pad: padding size
        stride: stride size
        k: kernel size
        in_ch: num of input chanels
    """
    if (S + pad - k // 2) % stride != 0:
        raise ValueError("(S + pad - k // 2) should be devidable by stride size")

    return ( 
        ((S + pad - k // 2) // stride) ** 2 # number of pool applications per 1 channel in 1 item
    ) * (
        (k ** 2) - 1 # number of flops per one pool (k-1 comparisons)
    ) * in_ch * B # repeat per each item in batch and per each channel


def flops_per_relu(S: int, B: int, in_ch: int) -> int:
    """
    Calc flops per ReLU layer

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        in_ch: num of input chanels
    """
    return ( 
        in_ch * S ** 2 # number of values in one item (1 comparison with 0 per value)
    ) * B # repeat per each item in batch


def flops_per_avg_pool(S: int, B: int, in_ch: int) -> int:
    """
    Calc flops per AvgPool2d layer

    Args:
        S: input shape (w & h, only w=h supported)
        B: batch size
        in_ch: num of input chanels
    """
    return ( 
        S ** 2 # number of flops in one channel (sum all (s^2 - 1) and divide (1))
    ) * in_ch * B # repeat per each channel in each item in batch


def flops_per_linear(B: int, in_ch: int, out_ch: int) -> int:
    """
    Calc flops per Linear layer (with bias)

    Args:
        B: batch size
        in_ch: num of input chanels
        out_ch: num of output chanels
    """
    return ( 
        2 * in_ch # number of flops per one scalar product ((2 * in_ch - 2) + 1 flop per bias)
    ) * out_ch * B # repeat per item in batch
