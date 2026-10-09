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


GPUS = 8 # paper's runs use 8 GPUs (--gpu=all), each with batch_size 125 from the prototxt
BATCH = 1000 # total batch per iteration, 8 x 125
GHOST = BATCH // GPUS
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
TEST_INTERVAL = 100 # solver test_interval
TEST_IMAGES = 200 * 125 # solver test_iter x test batch_size, more than the 10k test set, Caffe wraps around it
PRINT_INTERVAL = 1000 # iterations between progress lines, every test is in the CSV
SEED = 0
DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
FIELDS = ["iteration", "epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "time"]

# ResNet-56 on CIFAR-10 runs from the paper, solver parameters as in lrRangeSolver.prototxt, clrsolver.prototxt and solver.prototxt
# from github.com/lnsmith54/super-convergence, BN moving_average_fraction (maf) as in Table 1 and x.sh
RUNS = {
    # Fig. 2b, "Max Iter=5k": LR range test, triangular policy with base_lr 0 and stepsize = max_iter, so LR grows linearly 0 -> 3
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
    Learning rate of iteration it, as SGDSolver::GetLearningRate() computes it in Caffe for the run's lr_policy

    Args:
        run: run config from RUNS
        it: iteration, 0-based
    """
    if run["lr_policy"] == "triangular":
        # Policy from Smith's CLR paper: linear base_lr -> max_lr over stepsize iterations, then back, cycle = 2 x stepsize
        cycle = it // (2 * run["stepsize"])
        x = abs(it / run["stepsize"] - 2 * cycle - 1)
        return run["base_lr"] + (run["max_lr"] - run["base_lr"]) * max(0.0, 1 - x)
    # multistep: base_lr x gamma^(number of stepvalues passed)
    return run["base_lr"] * run["gamma"] ** sum(it >= s for s in run["stepvalue"])


class CaffeSGD(torch.optim.Optimizer):
    """
    SGD with momentum and L2 weight decay as Caffe's SGDSolver applies it: history = momentum * history + lr * (grad + weight_decay * w), w -= history
    PyTorch SGD keeps lr out of the history (w -= lr * history), which differs from Caffe whenever lr changes between iterations
    """

    def __init__(self, param_groups: list[dict], momentum: float, weight_decay: float) -> None:
        """
        Construct optimizer

        Args:
            param_groups: parameter groups, "lr" of a group is Caffe's lr_mult, LambdaLR multiplies it by the solver's rate
            momentum: solver momentum
            weight_decay: solver weight_decay
        """
        super().__init__(param_groups, {"momentum": momentum, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure) -> torch.Tensor:
        """
        Run forward and backward via closure (Lightning passes training_step + backward), then update parameters

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
    Load CIFAR-10 split as uint8 images (N, 3, 32, 32) and labels (N,)
    Order is data_batch_1..5, the same as keys of the LMDB that Caffe's convert_cifar_data writes

    Args:
        train: train or test split
    """
    ds = CIFAR10(DATA, train=train, download=True)
    return torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds.targets)


def train_batches(n: int, rank: int, world: int) -> list[list[int]]:
    """
    Construct image indices of one epoch for one process, one list per iteration, as Caffe's multi-GPU Data layer feeds them
    The LMDB is read in order without shuffling every epoch (shuffle exists only for ImageData layer),
    the reader deals images to the 8 GPUs round-robin, so GPU g gets images g, g + 8, ... of each 1000 (one BN chunk)
    With several processes each takes 8 / world consecutive chunks of every batch

    Args:
        n: training set size, multiple of BATCH
        rank: process index
        world: number of processes
    """
    batches = torch.arange(n).view(n // BATCH, BATCH // GPUS, GPUS).transpose(1, 2).reshape(n // BATCH, BATCH)
    return batches.tensor_split(world, dim=1)[rank].tolist()


class TestBatches:
    """
    Image indices of one Caffe test: test_iter x batch_size images read from where the previous test stopped, wrapping around the test set
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
        self.cursor = TEST_IMAGES % n # Caffe also tests at iteration 0, that pass is skipped but its images are read

    def __iter__(self):
        """
        Yield this process's share of batches of BATCH images instead of 125, BN in test uses running stats, so outputs don't depend on batching
        """
        idx = (self.cursor + torch.arange(TEST_IMAGES)) % self.n
        self.cursor = (self.cursor + TEST_IMAGES) % self.n
        return iter([b.tensor_split(self.world)[self.rank] for b in idx.split(BATCH)])

    def __len__(self) -> int:
        """
        Number of batches per test, the same in every process
        """
        return math.ceil(TEST_IMAGES / BATCH)


class SuperConvergence(L.LightningModule):
    """
    Train ResNet-56 with Caffe's SGD and LR policy from the paper, write one CSV row per test
    On several GPUs runs as DDP, one process per GPU, and every iteration computes the same as on one GPU up to float rounding:
    processes split each batch by BN chunks, DDP averages gradients like Caffe's P2PSync, tests use the BN stats of process 0
    """

    def __init__(self, cfg: dict, mean: torch.Tensor, train_set: TensorDataset, test_set: TensorDataset, out: Path) -> None:
        """
        Construct model

        Args:
            cfg: run config from RUNS
            mean: mean training image (3, 32, 32), 0-255 scale
            train_set: uint8 training images and labels
            test_set: uint8 test images and labels
            out: CSV path, the JSON goes next to it
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
        Convert uint8 batch to network input like the prototxt's transform_param: raw 0-255 pixels minus mean image, no scale

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
        x = self.preprocess(x)
        # mirror: true is the only augmentation, crop_size equals image size, so there are no random crops
        # Flips are drawn for the whole batch and split like the images, so any number of GPUs gets the same ones
        flip = torch.rand(BATCH, generator=self.flips).tensor_split(self.trainer.world_size)[self.global_rank] < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1).to(x.device), x.flip(3), x)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y) # SoftmaxWithLoss averages per GPU, the root solver averages the 8 GPUs, DDP the processes
        self.train_sums[0] += loss.detach() * len(y)
        self.train_sums[1] += (logits.argmax(1) == y).sum()
        self.train_sums[2] += len(y)
        self.lr = self.trainer.optimizers[0].param_groups[0]["lr"] # scheduler steps after the optimizer, so this is this iteration's LR
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Accumulate test loss and accuracy of one batch, BN uses running stats like Caffe's use_global_stats: true

        Args:
            batch: uint8 images and labels
            batch_idx: batch index in test
        """
        x, y = batch
        logits = self.model(self.preprocess(x))
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
        Construct loader of this process's share of every test, the sampler keeps its cursor between tests
        """
        sampler = TestBatches(len(self.test_set), self.global_rank, self.trainer.world_size)
        return DataLoader(self.test_set, sampler=sampler, batch_size=None, pin_memory=self.device.type == "cuda")

    def on_train_start(self) -> None:
        """
        Start wall clock and metric accumulators, write config and environment to JSON and CSV header
        """
        self.start = time.perf_counter()
        # Loss sum, correct, seen since last test, kept on GPU to avoid a sync every iteration
        # Not buffers: DDP would overwrite them with process 0's values at the first forward after training
        self.train_sums = torch.zeros(3, device=self.device)
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
        print(self.out.stem, json.dumps({"config": self.cfg, "env": env}), flush=True)
        self.out.with_suffix(".json").write_text(json.dumps({"config": self.cfg, "env": env}, indent=2))
        with open(self.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def on_validation_epoch_start(self) -> None:
        """
        Copy BN running stats of process 0 to the others before a test
        Caffe tests with the root GPU's stats, which come from chunk 0 only. DDP broadcasts process 0's buffers before
        every training forward, but then each other process updates its copy from its own first chunk
        """
        if self.trainer.world_size > 1:
            for b in self.model.buffers():
                torch.distributed.broadcast(b, src=0)

    def on_validation_epoch_end(self) -> None:
        """
        Append train metrics since last test and test metrics, summed over processes, to CSV, file is complete after every row
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
        Construct Caffe SGD with the prototxt's lr_mult per parameter and the solver's LR policy stepped every iteration
        """
        fc_bias = self.model[-1].bias
        groups = [
            # weight_decay applies to all of these: convs and FC (decay_mult 1), Scale layers (no param block, defaults to 1)
            {"params": [p for p in self.model.parameters() if p is not fc_bias], "lr": 1.0},
            {"params": [fc_bias], "lr": 2.0}, # post_FC bias: lr_mult 2, decay_mult not set, so 1
        ]
        optimizer = CaffeSGD(groups, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = LambdaLR(optimizer, lambda it: caffe_lr(self.cfg, it))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def main() -> None:
    """
    Train selected runs on all visible GPUs, write results/<run>.csv (one row per test) and results/<run>.json (config and environment)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", choices=list(RUNS))
    parser.add_argument("--max-hours", type=float,
                        help="stop when this many hours have passed since start, the run in progress ends early, for sessions with a time limit (Kaggle: 12 h)")
    args = parser.parse_args()
    deadline = time.monotonic() + args.max_hours * 3600 if args.max_hours else None

    torch.backends.cudnn.allow_tf32 = False # so that FP32 means FP32 on Ampere and newer GPUs
    torch.backends.cuda.matmul.allow_tf32 = False
    RESULTS.mkdir(exist_ok=True)
    devices = max(torch.cuda.device_count(), 1)
    if GPUS % devices:
        raise SystemExit(f"{devices} GPUs can't share the paper's {GPUS} BN chunks evenly, select 1, 2, 4 or 8 with CUDA_VISIBLE_DEVICES")

    x_train, y_train = load_cifar(train=True)
    x_test, y_test = load_cifar(train=False)
    mean = x_train.float().mean(0) # mean.binaryproto of the full training set, Table 1 runs only swap the train LMDB
    for name in args.runs:
        cfg = RUNS[name]
        n = cfg["train_size"] # first n images, the paper doesn't say which ones its smaller LMDBs hold
        L.seed_everything(SEED)
        module = SuperConvergence(cfg, mean, TensorDataset(x_train[:n], y_train[:n]), TensorDataset(x_test, y_test), RESULTS / f"{name}.csv")
        trainer = L.Trainer(
            max_steps=cfg["max_iter"],
            max_epochs=-1, # length is set in iterations like in Caffe
            max_time=timedelta(seconds=deadline - time.monotonic()) if deadline else None, # Lightning tests once more at the stop
            devices=devices, # more than 1 starts DDP, Lightning reruns this script for every other GPU
            use_distributed_sampler=False, # train_batches and TestBatches split batches between processes themselves
            val_check_interval=TEST_INTERVAL,
            check_val_every_n_epoch=None, # count val_check_interval in iterations across epochs
            num_sanity_val_steps=0,
            precision="32-true", # paper trains in FP32
            benchmark=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False, # one bar per epoch floods notebook output, on_validation_epoch_end prints progress
        )
        trainer.fit(module)


if __name__ == "__main__":
    main()
