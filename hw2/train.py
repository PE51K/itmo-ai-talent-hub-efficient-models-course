import argparse
import csv
import json
import math
import platform
import time
from datetime import timedelta
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

from models import get_model


GPUS = 8 # paper's GPUs, BN normalizes each GPU's share of the batch
BATCH = 1000 # images per iteration, 8 x 125
GHOST = BATCH // GPUS
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
TEST_INTERVAL = 100 # iterations between tests
TEST_IMAGES = 200 * 125 # test_iter x batch_size, more than the 10k test set, Caffe wraps around it
PRINT_INTERVAL = 1000
SEED = 0
DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
FIELDS = ["iteration", "epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "time"]

# ResNet-56 on CIFAR-10 runs from the paper, solver parameters from github.com/lnsmith54/super-convergence, BN MAF from Table 1
RUNS = {
    # Fig. 2b, "Max Iter=5k": LR range test, LR grows linearly 0 -> 3
    "lr_range_test": {"lr_policy": "triangular", "base_lr": 0.0, "max_lr": 3.0, "stepsize": 5_000, "max_iter": 5_000, "maf": 0.95, "train_size": 50_000},
    # Fig. 1a: super-convergence, one CLR cycle 0.1 -> 3 -> 0.1 (92.4 %)
    "clr": {"lr_policy": "triangular", "base_lr": 0.1, "max_lr": 3.0, "stepsize": 5_000, "max_iter": 10_000, "maf": 0.95, "train_size": 50_000},
    # Fig. 1a: typical training, PC-LR 0.35 divided by 10 at iterations 50k and 70k (91.2 %)
    "pc_lr": {"lr_policy": "multistep", "base_lr": 0.35, "gamma": 0.1, "stepvalue": [50_000, 70_000], "max_iter": 80_000, "maf": 0.999, "train_size": 50_000},
    # Table 1: same pair on 10,000 training samples (80.6 % and 71.4 %)
    "clr_10k": {"lr_policy": "triangular", "base_lr": 0.1, "max_lr": 3.0, "stepsize": 5_000, "max_iter": 10_000, "maf": 0.95, "train_size": 10_000},
    "pc_lr_10k": {"lr_policy": "multistep", "base_lr": 0.35, "gamma": 0.1, "stepvalue": [50_000, 70_000], "max_iter": 80_000, "maf": 0.999, "train_size": 10_000},
}


def caffe_lr(run: dict, it: int) -> float:
    """
    Calc LR of iteration it as Caffe's triangular or multistep policy

    Args:
        run: run config from RUNS
        it: iteration, 0-based
    """
    if run["lr_policy"] == "triangular":
        cycle = it // (2 * run["stepsize"])
        x = abs(it / run["stepsize"] - 2 * cycle - 1)
        return run["base_lr"] + (run["max_lr"] - run["base_lr"]) * max(0.0, 1 - x)
    return run["base_lr"] * run["gamma"] ** sum(it >= s for s in run["stepvalue"])


class CaffeSGD(torch.optim.Optimizer):
    """
    Caffe's SGD: history = momentum * history + lr * (grad + weight_decay * w), w -= history
    Unlike PyTorch's SGD, LR is inside the history, which matters when LR changes every iteration
    """

    def __init__(self, param_groups: list[dict], momentum: float, weight_decay: float) -> None:
        """
        Construct optimizer

        Args:
            param_groups: parameter groups, "lr" of a group is Caffe's lr_mult, LambdaLR multiplies it by the policy's LR
            momentum: solver momentum
            weight_decay: solver weight_decay
        """
        super().__init__(param_groups, {"momentum": momentum, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure) -> torch.Tensor:
        """
        Run forward and backward via closure, then update parameters

        Args:
            closure: computes loss and gradients
        """
        with torch.enable_grad():
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                history = self.state[p].setdefault("history", torch.zeros_like(p))
                history.mul_(group["momentum"]).add_(p.grad + group["weight_decay"] * p, alpha=group["lr"])
                p.sub_(history)
        return loss


def load_cifar(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load CIFAR-10 split as uint8 images (N, 3, 32, 32) and labels (N,), in the order of Caffe's CIFAR-10 LMDB

    Args:
        train: train or test split
    """
    ds = CIFAR10(DATA, train=train, download=True)
    return torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds.targets)


def train_batches(n: int, rank: int, world: int) -> list[list[int]]:
    """
    Construct image indices of one epoch for one process, one list per iteration
    Caffe reads the LMDB in order without shuffling and deals images to the 8 GPUs round-robin,
    so GPU g gets images g, g + 8, ... of a batch, each process takes 8 / world of these BN chunks

    Args:
        n: training set size, multiple of BATCH
        rank: process index
        world: number of processes
    """
    batches = torch.arange(n).view(n // BATCH, BATCH // GPUS, GPUS).transpose(1, 2).reshape(n // BATCH, BATCH)
    return batches.tensor_split(world, dim=1)[rank].tolist()


class TestBatches:
    """
    Image indices of one test for one process: TEST_IMAGES images from where the previous test stopped, wrapping around the test set
    """

    def __init__(self, n: int, rank: int, world: int) -> None:
        """
        Construct cursor

        Args:
            n: test set size
            rank: process index
            world: number of processes
        """
        self.n = n
        self.rank = rank
        self.world = world
        self.cursor = TEST_IMAGES % n # Caffe's test at iteration 0 is skipped, but its images are read

    def __iter__(self):
        """
        Yield this process's share of every batch, BN in test uses running stats, so batching doesn't change outputs
        """
        idx = (self.cursor + torch.arange(TEST_IMAGES)) % self.n
        self.cursor = (self.cursor + TEST_IMAGES) % self.n
        return iter([b.tensor_split(self.world)[self.rank] for b in idx.split(BATCH)])

    def __len__(self) -> int:
        """
        Number of batches per test
        """
        return math.ceil(TEST_IMAGES / BATCH)


class SuperConvergence(L.LightningModule):
    """
    Train ResNet-56 as the paper's Caffe runs, one process per GPU, write one CSV row per test
    """

    def __init__(self, cfg: dict, mean: torch.Tensor, train_set: TensorDataset, test_set: TensorDataset, out: Path) -> None:
        """
        Construct model

        Args:
            cfg: run config from RUNS
            mean: mean training image (3, 32, 32), 0-255 scale
            train_set: uint8 training images and labels
            test_set: uint8 test images and labels
            out: CSV path, JSON goes next to it
        """
        super().__init__()
        self.cfg = cfg
        self.train_set = train_set
        self.test_set = test_set
        self.out = out
        self.model = get_model(cfg["maf"], GHOST)
        self.flips = torch.Generator().manual_seed(SEED)
        self.register_buffer("mean", mean)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert uint8 batch to network input as in the prototxt: raw 0-255 pixels minus mean image

        Args:
            x: uint8 images (B, 3, 32, 32)
        """
        return x.float() - self.mean

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Run forward of one iteration, return loss for Lightning to backprop

        Args:
            batch: uint8 images and labels
            batch_idx: batch index in epoch
        """
        x, y = batch
        # Random mirror, the only augmentation, drawn for the whole batch so that any number of GPUs gets the same flips
        flip = torch.rand(BATCH, generator=self.flips).tensor_split(self.trainer.world_size)[self.global_rank] < 0.5
        x = self.preprocess(x)
        x = torch.where(flip.view(-1, 1, 1, 1).to(x.device), x.flip(3), x)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.train_sums[0] += loss.detach() * len(y)
        self.train_sums[1] += (logits.argmax(1) == y).sum()
        self.train_sums[2] += len(y)
        self.lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Accumulate test loss and accuracy of one batch

        Args:
            batch: uint8 images and labels
            batch_idx: batch index in test
        """
        x, y = batch
        logits = self.model(self.preprocess(x)) # DDP copies rank 0's buffers before every forward, all processes test with its BN stats
        self.test_sums[0] += F.cross_entropy(logits, y, reduction="sum")
        self.test_sums[1] += (logits.argmax(1) == y).sum()
        self.test_sums[2] += len(y)

    def train_dataloader(self) -> DataLoader:
        """
        Construct loader of this process's share of every training batch
        """
        sampler = train_batches(len(self.train_set), self.global_rank, self.trainer.world_size)
        return DataLoader(self.train_set, sampler=sampler, batch_size=None, pin_memory=self.device.type == "cuda")

    def val_dataloader(self) -> DataLoader:
        """
        Construct loader of this process's share of every test
        """
        sampler = TestBatches(len(self.test_set), self.global_rank, self.trainer.world_size)
        return DataLoader(self.test_set, sampler=sampler, batch_size=None, pin_memory=self.device.type == "cuda")

    def on_train_start(self) -> None:
        """
        Start wall clock and metric sums, write config and environment to JSON and CSV header
        """
        self.start = time.perf_counter()
        self.train_sums = torch.zeros(3, device=self.device) # loss sum, correct, seen since last test
        self.test_sums = torch.zeros(3, device=self.device)
        if not self.trainer.is_global_zero:
            return
        cuda = self.device.type == "cuda"
        env = {
            "gpu": torch.cuda.get_device_name(self.device) if cuda else str(self.device),
            "devices": self.trainer.world_size,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if cuda else None,
            "torch": torch.__version__,
            "lightning": L.__version__,
            "python": platform.python_version(),
        }
        self.out.with_suffix(".json").write_text(json.dumps({"config": self.cfg, "env": env}, indent=2))
        with open(self.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def on_validation_epoch_end(self) -> None:
        """
        Append train metrics since last test and test metrics, summed over processes, to CSV
        """
        train = self.trainer.strategy.reduce(self.train_sums, reduce_op="sum").tolist()
        test = self.trainer.strategy.reduce(self.test_sums, reduce_op="sum").tolist()
        self.train_sums.zero_()
        self.test_sums.zero_()
        if not self.trainer.is_global_zero:
            return
        elapsed = time.perf_counter() - self.start
        it = self.global_step
        row = {
            "iteration": it,
            "epoch": it * BATCH / self.cfg["train_size"],
            "lr": self.lr, # LR of the last iteration
            "train_loss": train[0] / train[2],
            "train_acc": train[1] / train[2],
            "test_loss": test[0] / test[2],
            "test_acc": test[1] / test[2],
            "time": elapsed,
        }
        with open(self.out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        if it % PRINT_INTERVAL == 0 or it == self.cfg["max_iter"]:
            eta = elapsed / it * (self.cfg["max_iter"] - it)
            print(f"{self.out.stem} iteration {it}/{self.cfg['max_iter']} lr {row['lr']:.4f} train_acc {row['train_acc']:.4f} "
                  f"test_acc {row['test_acc']:.4f} elapsed {elapsed / 3600:.2f} h eta {eta / 3600:.2f} h", flush=True)

    def configure_optimizers(self) -> dict:
        """
        Construct Caffe SGD with the prototxt's lr_mult and the run's LR policy, stepped every iteration
        """
        fc_bias = self.model[-1].bias
        groups = [
            {"params": [p for p in self.model.parameters() if p is not fc_bias], "lr": 1.0},
            {"params": [fc_bias], "lr": 2.0},
        ]
        # Weight decay applies to all parameters, BN scale and shift included
        optimizer = CaffeSGD(groups, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = LambdaLR(optimizer, lambda it: caffe_lr(self.cfg, it))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def main() -> None:
    """
    Train selected runs on all GPUs, write results/<run>.csv (one row per test) and results/<run>.json (config and environment)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", choices=list(RUNS))
    parser.add_argument("--max-hours", type=float, help="stop training after this many hours, for all runs together")
    args = parser.parse_args()
    deadline = time.monotonic() + args.max_hours * 3600 if args.max_hours else None

    torch.backends.cudnn.allow_tf32 = False # so that FP32 means FP32 on Ampere and newer GPUs
    torch.backends.cuda.matmul.allow_tf32 = False
    RESULTS.mkdir(exist_ok=True)
    devices = max(torch.cuda.device_count(), 1)
    if GPUS % devices != 0:
        raise ValueError(f"number of GPUs should divide {GPUS}, got {devices}")

    x_train, y_train = load_cifar(train=True)
    x_test, y_test = load_cifar(train=False)
    mean = x_train.float().mean(0) # of all 50k images, also for the 10k runs
    for name in args.runs:
        cfg = RUNS[name]
        n = cfg["train_size"] # first n images, the paper doesn't say which
        L.seed_everything(SEED)
        module = SuperConvergence(cfg, mean, TensorDataset(x_train[:n], y_train[:n]), TensorDataset(x_test, y_test), RESULTS / f"{name}.csv")
        trainer = L.Trainer(
            max_steps=cfg["max_iter"],
            max_epochs=-1,
            max_time=timedelta(seconds=deadline - time.monotonic()) if deadline else None,
            devices=devices,
            use_distributed_sampler=False, # train_batches and TestBatches split batches between processes
            val_check_interval=TEST_INTERVAL,
            check_val_every_n_epoch=None,
            num_sanity_val_steps=0,
            precision="32-true",
            benchmark=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(module)


if __name__ == "__main__":
    main()
