import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaffeBatchNorm2d(nn.Module):
    """
    Caffe BatchNorm + Scale as in the paper's 8-GPU runs: each GPU normalizes its 125 images with their own stats,
    running stats come from the root GPU and are averaged with weight maf^age, test uses running stats
    """

    def __init__(self, num_features: int, maf: float, ghost: int) -> None:
        """
        Construct BN

        Args:
            num_features: number of channels
            maf: Caffe moving_average_fraction
            ghost: images per GPU
        """
        super().__init__()
        self.maf = maf
        self.ghost = ghost
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("mean_sum", torch.zeros(num_features))
        self.register_buffer("var_sum", torch.zeros(num_features))
        self.register_buffer("count", torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize every chunk of `ghost` images with its batch stats in training, with running stats in eval

        Args:
            x: input (B, C, S, S)
        """
        if not self.training:
            return F.batch_norm(x, self.mean_sum / self.count, self.var_sum / self.count, self.weight, self.bias, training=False)
        with torch.no_grad():
            first = x[:self.ghost]
            self.count.mul_(self.maf).add_(1)
            self.mean_sum.mul_(self.maf).add_(first.mean((0, 2, 3)))
            self.var_sum.mul_(self.maf).add_(first.var((0, 2, 3))) # unbiased, as Caffe
        return torch.cat([F.batch_norm(c, None, None, self.weight, self.bias, training=True) for c in x.split(self.ghost)])


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
            ghost: images per GPU
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
        self.bn1 = CaffeBatchNorm2d(channels, maf, ghost)
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = CaffeBatchNorm2d(channels, maf, ghost)
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
        ghost: images per GPU
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
        CaffeBatchNorm2d(16, maf, ghost),
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
