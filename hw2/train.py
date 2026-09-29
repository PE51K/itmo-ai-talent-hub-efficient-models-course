import argparse
import csv
import json
import platform
import time
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, OneCycleLR
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torchvision.datasets import CIFAR10

from models import get_model


BATCH = 1000 # paper: 8 GPUs x 125
GHOST = 125 # BN batch statistics per GPU in the paper
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
EVALS = 100 # test set evaluations per run, run lengths are rounded to a multiple of it
SEED = 0
DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
FIELDS = ["step", "epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "time"]

# ResNet-56 on CIFAR-10 runs from the paper, steps (iterations) are the paper's at --scale 1
# Solvers are clrsolver.prototxt, solver.prototxt and lrRangeSolver.prototxt from github.com/lnsmith54/super-convergence
RUNS = {
    # Fig. 2b: LR range test, LR grows linearly from 0 to max_lr
    "range": {"policy": "range", "max_lr": 3.0, "steps": 5_000, "maf": 0.95, "train_size": 50_000},
    # Fig. 1a: one CLR cycle 0.1 -> 3 -> 0.1 (92.4 %) vs piecewise constant LR 0.35 with 10x drops (91.2 %)
    "clr": {"policy": "clr", "base_lr": 0.1, "max_lr": 3.0, "steps": 10_000, "maf": 0.95, "train_size": 50_000},
    "pc": {"policy": "pc", "max_lr": 0.35, "milestones": [50_000, 70_000], "steps": 80_000, "maf": 0.999, "train_size": 50_000},
    # Table 1: same pair on 10k training images (80.6 % vs 71.4 %)
    "clr_10k": {"policy": "clr", "base_lr": 0.1, "max_lr": 3.0, "steps": 10_000, "maf": 0.95, "train_size": 10_000},
    "pc_10k": {"policy": "pc", "max_lr": 0.35, "milestones": [50_000, 70_000], "steps": 80_000, "maf": 0.999, "train_size": 10_000},
}


def scale_run(run: dict, scale: float) -> dict:
    """
    Shorten run: scale steps and LR milestones, LR values and BN maf stay as in the paper

    Args:
        run: run config from RUNS
        scale: multiplier of run length, 1 = paper
    """
    scaled = dict(run, steps=max(1, round(run["steps"] * scale / EVALS)) * EVALS)
    if "milestones" in run:
        scaled["milestones"] = [round(m * scale) for m in run["milestones"]]
    return scaled


def load_cifar(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load CIFAR-10 split as uint8 images (N, 3, 32, 32) and labels (N,)

    Args:
        train: train or test split
    """
    ds = CIFAR10(DATA, train=train, download=True)
    return torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds.targets)


def loader(x: torch.Tensor, y: torch.Tensor, shuffle: bool) -> DataLoader:
    """
    Construct loader that takes a whole batch with one tensor index, per-image __getitem__ + collate is slow on Colab's 2 CPUs

    Args:
        x: uint8 images
        y: labels
        shuffle: reshuffle every epoch and drop incomplete last batch (training)
    """
    ds = TensorDataset(x, y)
    sampler = BatchSampler(RandomSampler(ds) if shuffle else SequentialSampler(ds), batch_size=BATCH, drop_last=shuffle)
    return DataLoader(ds, sampler=sampler, batch_size=None, pin_memory=torch.cuda.is_available())


class SuperConvergence(L.LightningModule):
    """
    Train ResNet-56 with SGD and per-iteration LR policy from the paper, write one CSV row per test set evaluation
    """

    def __init__(self, cfg: dict, mean: torch.Tensor, out: Path) -> None:
        """
        Construct model and metric accumulators

        Args:
            cfg: run config from RUNS, already scaled
            mean: mean training image (3, 32, 32), 0-255 scale
            out: CSV path
        """
        super().__init__()
        self.cfg = cfg
        self.out = out
        self.model = get_model(cfg["maf"], GHOST).to(memory_format=torch.channels_last) # NHWC is what FP16 tensor core convs want
        self.register_buffer("mean", mean)
        # Loss sum, correct, seen since last evaluation, kept on GPU to avoid a sync every step
        self.register_buffer("train_sums", torch.zeros(3), persistent=False)
        self.register_buffer("test_sums", torch.zeros(3), persistent=False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert uint8 batch to network input like Caffe's cifar10 example: raw 0-255 pixels minus mean image, no std scaling

        Args:
            x: uint8 images (B, 3, 32, 32)
        """
        return (x.float() - self.mean).contiguous(memory_format=torch.channels_last)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Run one SGD iteration, return loss for Lightning to backprop

        Args:
            batch: uint8 images and labels
            batch_idx: batch index in epoch
        """
        x, y = batch
        x = self.preprocess(x)
        # mirror: true is the only augmentation, crop_size equals image size, so there are no random crops
        flip = torch.rand(len(x), 1, 1, 1, device=x.device) < 0.5
        x = torch.where(flip, x.flip(3), x)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
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
            batch_idx: batch index in test set
        """
        x, y = batch
        logits = self.model(self.preprocess(x))
        self.test_sums[0] += F.cross_entropy(logits, y, reduction="sum")
        self.test_sums[1] += (logits.argmax(1) == y).sum()
        self.test_sums[2] += len(y)

    def on_train_start(self) -> None:
        """
        Start wall clock and write CSV header
        """
        self.start = time.perf_counter()
        with open(self.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def on_validation_epoch_end(self) -> None:
        """
        Append train metrics since last evaluation and test metrics to CSV, file is complete after every row
        """
        train, test = self.train_sums.tolist(), self.test_sums.tolist()
        row = {
            "step": self.global_step,
            "epoch": self.global_step * BATCH / self.cfg["train_size"],
            "lr": self.lr, # LR of the last iteration, OneCycleLR extrapolates below 0 after the last one
            "train_loss": train[0] / train[2],
            "train_acc": train[1] / train[2],
            "test_loss": test[0] / test[2],
            "test_acc": test[1] / test[2],
            "time": time.perf_counter() - self.start,
        }
        with open(self.out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"{self.out.stem} step {row['step']}/{self.cfg['steps']} lr {row['lr']:.4f} train_acc {row['train_acc']:.4f} "
              f"test_acc {row['test_acc']:.4f} {row['time'] / 60:.1f} min", flush=True)
        self.train_sums.zero_()
        self.test_sums.zero_()

    def configure_optimizers(self) -> dict:
        """
        Construct SGD as in the paper's solvers and LR policy stepped every iteration
        """
        cfg = self.cfg
        # Decay on all params incl. BN scale/shift, Caffe Scale layers decay by default too
        optimizer = torch.optim.SGD(self.parameters(), lr=cfg["max_lr"], momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        if cfg["policy"] == "clr":
            # Caffe "triangular" policy, one cycle (stepsize = steps / 2): linear base_lr -> max_lr -> base_lr
            scheduler = OneCycleLR(
                optimizer,
                max_lr=cfg["max_lr"],
                total_steps=cfg["steps"],
                pct_start=0.5,
                anneal_strategy="linear",
                cycle_momentum=False, # paper's ResNet-56 runs keep momentum 0.9
                div_factor=cfg["max_lr"] / cfg["base_lr"], # start LR = max_lr / div_factor
                final_div_factor=1.0, # end LR = start LR / final_div_factor
            )
        elif cfg["policy"] == "range":
            scheduler = LambdaLR(optimizer, lambda step: step / cfg["steps"])
        else:
            scheduler = MultiStepLR(optimizer, milestones=cfg["milestones"], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def main() -> None:
    """
    Train selected runs, write results/<run>.csv, results/runs.json (configs as trained) and results/env.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", choices=list(RUNS))
    parser.add_argument("--scale", type=float, default=1.0, help="multiplier of run lengths and LR milestones, 1 = paper")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    cuda = torch.cuda.is_available()
    env = {
        "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "torch": torch.__version__,
        "lightning": L.__version__,
        "python": platform.python_version(),
    }
    (RESULTS / "env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2))

    runs_path = RESULTS / "runs.json"
    trained = json.loads(runs_path.read_text()) if runs_path.exists() else {}
    x_train, y_train = load_cifar(train=True)
    x_test, y_test = load_cifar(train=False)
    for name in args.runs:
        cfg = scale_run(RUNS[name], args.scale)
        print(name, cfg, flush=True)
        n = cfg["train_size"] # first n images, CIFAR-10 training batches are already in random order
        L.seed_everything(SEED)
        module = SuperConvergence(cfg, x_train[:n].float().mean(0), RESULTS / f"{name}.csv")
        trainer = L.Trainer(
            max_steps=cfg["steps"],
            max_epochs=-1, # length is set in iterations like in Caffe
            val_check_interval=cfg["steps"] // EVALS,
            check_val_every_n_epoch=None, # count val_check_interval in iterations across epochs
            num_sanity_val_steps=0,
            precision="16-mixed" if cuda else "32-true", # paper trains in FP32, FP16 tensor cores make T4 runs affordable
            benchmark=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False, # one bar per epoch floods notebook output, on_validation_epoch_end prints progress
        )
        trainer.fit(module, loader(x_train[:n], y_train[:n], shuffle=True), loader(x_test, y_test, shuffle=False))
        trained[name] = cfg
        runs_path.write_text(json.dumps(trained, indent=2))


if __name__ == "__main__":
    main()
