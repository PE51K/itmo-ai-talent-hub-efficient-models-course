import torch
import torch.nn as nn


def get_model(S: int, device: torch.device | str = "cpu") -> nn.Sequential:
    """
    Construct model from hw1 as nn.Sequential

    Args:
        S: input shape (w & h, only w=h supported)
        device: cpu or cuda
    """
    # Check that shape is divisable by 16
    if S % 16 != 0:
        raise ValueError("S should be divisable by 16")
    # Construct and return model
    return nn.Sequential( # -> (B, 3, S, S)
        nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        ), # -> (B, 32, S/2, S/2)
        nn.ReLU(
            inplace=True,
        ), # -> (B, 32, S/2, S/2)
        nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        ), # -> (B, 32, S/4, S/4)
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        ), # -> (B, 64, S/4, S/4)
        nn.ReLU(
            inplace=True,
        ), # -> (B, 64, S/4, S/4)
        nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ), # -> (B, 128, S/8, S/8)
        nn.ReLU(
            inplace=True,
        ), # -> (B, 128, S/8, S/8)
        nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ), # -> (B, 256, S/8, S/8)
        nn.ReLU(
            inplace=True,
        ), # -> (B, 256, S/8, S/8)
        nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ), # -> (B, 256, S/16, S/16)
        nn.ReLU(
            inplace=True,
        ), # -> (B, 256, S/16, S/16)
        nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ), # -> (B, 512, S/16, S/16)
        nn.ReLU(
            inplace=True,
        ), # -> (B, 512, S/16, S/16)
        # Head
        nn.AvgPool2d(
            kernel_size=S//16,
            stride=1,
            padding=0,
        ), # -> (B, 512, 1, 1)
        nn.Flatten(), # -> (B, 512)
        nn.Linear(
            in_features= 512,
            out_features=256,
        ), # -> (B, 256)
        nn.ReLU(
            inplace=True,
        ), # -> (B, 256)
        nn.Linear(
            in_features= 256,
            out_features=100,
        ), # -> (B, 100)
    ).to( # Transfer to device
        device=device
    )
