import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostBatchNorm2d(nn.BatchNorm2d):
    """
    Caffe BatchNorm + Scale of the paper's 8-GPU runs: batch stats per chunk of `ghost` images (one GPU's share),
    running stats from the first chunk only (root GPU)
    """

    def __init__(self, num_features: int, maf: float, ghost: int) -> None:
        """
        Construct BN

        Args:
            num_features: number of channels
            maf: Caffe moving_average_fraction, weight of the old running stats
            ghost: images per chunk
        """
        super().__init__(
            num_features=num_features,
            momentum=1 - maf, # weight of the new batch stats
        )
        self.maf = maf
        self.ghost = ghost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize every chunk with its own batch stats in training, with running stats in eval

        Args:
            x: input (B, C, S, S)
        """
        if self.training:
            first, *rest = x.split(self.ghost)
            return torch.cat([
                super().forward(first), # updates running stats
                *[F.batch_norm(c, None, None, self.weight, self.bias, training=True, eps=self.eps) for c in rest],
            ])
        # Caffe averages batch stats only, PyTorch's EMA also keeps maf^t of the initial mean 0 / var 1
        decay = self.maf ** self.num_batches_tracked
        mean = self.running_mean / (1 - decay)
        var = (self.running_var - decay) / (1 - decay)
        return F.batch_norm(x, mean, var, self.weight, self.bias, training=False, eps=self.eps)


class Block(nn.Module):
    """
    Residual block of Caffe ResNet-56 from the paper (Tables 3, 4): conv-BN-ReLU-conv-BN + shortcut -> ReLU
    Downsampling block halves S with stride 2, shortcut is 3x3/2 average pooling, output is zero padded to 2x channels
    """

    def __init__(self, channels: int, maf: float, ghost: int, downsample: bool = False) -> None:
        """
        Construct block layers

        Args:
            channels: numChannels, width of both convs
            maf: Caffe BN moving_average_fraction
            ghost: images per BN chunk
            downsample: halve S and double channels
        """
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2 if downsample else 1,
            padding=1,
            bias=False, # BN right after cancels Caffe's conv bias
        )
        self.bn1 = GhostBatchNorm2d(channels, maf, ghost)
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = GhostBatchNorm2d(channels, maf, ghost)
        # Caffe rounds pooling output size up: 32 -> 16, as the stride 2 conv
        self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run block

        Args:
            x: input (B, C, S, S)
        """
        out = F.relu(self.bn1(self.conv1(x))) # -> (B, C, S/stride, S/stride)
        out = F.relu(self.bn2(self.conv2(out)) + self.shortcut(x)) # -> (B, C, S/stride, S/stride)
        if self.downsample:
            out = torch.cat([out, torch.zeros_like(out)], dim=1) # DummyData zeros + Concat -> (B, 2C, S/2, S/2)
        return out


def get_model(maf: float, ghost: int, device: torch.device | str = "cpu") -> nn.Sequential:
    """
    Construct Caffe ResNet-56 for CIFAR-10 from the paper (Table 5, Resnet56Cifar.prototxt) as nn.Sequential

    Args:
        maf: Caffe BN moving_average_fraction
        ghost: images per BN chunk, per-GPU batch in the paper
        device: cpu or cuda
    """
    model = nn.Sequential( # -> (B, 3, 32, 32)
        nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1, # Table 5 says 2, the prototxt has 1
            padding=1,
            bias=False,
        ), # -> (B, 16, 32, 32)
        GhostBatchNorm2d(16, maf, ghost),
        nn.ReLU(
            inplace=True,
        ), # -> (B, 16, 32, 32)
        *[Block(16, maf, ghost) for _ in range(9)], # -> (B, 16, 32, 32)
        Block(16, maf, ghost, downsample=True), # -> (B, 32, 16, 16)
        *[Block(32, maf, ghost) for _ in range(8)], # -> (B, 32, 16, 16)
        Block(32, maf, ghost, downsample=True), # -> (B, 64, 8, 8)
        *[Block(64, maf, ghost) for _ in range(8)], # -> (B, 64, 8, 8)
        # Head
        nn.AvgPool2d(
            kernel_size=8,
            stride=1,
        ), # -> (B, 64, 1, 1)
        nn.Flatten(), # -> (B, 64)
        nn.Linear(
            in_features=64,
            out_features=10,
        ), # -> (B, 10)
    )
    # Caffe fillers: msra for convs, xavier for FC (uniform, bound = sqrt(3 / fan_in)), zero biases
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            bound = math.sqrt(3 / m.in_features)
            nn.init.uniform_(m.weight, -bound, bound)
            nn.init.zeros_(m.bias)
    return model.to( # Transfer to device
        device=device
    )
