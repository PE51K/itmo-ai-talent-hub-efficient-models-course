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
    return (
        ((S + pad - k // 2) // stride) ** 2 # number of kernel applications per 1 kernel in 1 item
    ) * (
        2 * (in_ch * k ** 2) # number of flops per kernel (in_ch * k^2 MACs, 1 MAC = 2 flops)
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


def flops(image_size, batch):
    """
    Calc flops per one forward pass of model from models.py

    Args:
        image_size: input shape (w & h, only w=h supported), int or np.ndarray
        batch: batch size, int or np.ndarray
    """
    S, B = image_size, batch
    return (
        flops_per_conv(S, B, pad=3, stride=2, k=7, in_ch=3, out_ch=32) # -> S/2
        + fllops_per_max_pool(S // 2, B, pad=1, stride=2, k=3, in_ch=32) # -> S/4
        + flops_per_relu(S // 4, B, in_ch=32)
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
