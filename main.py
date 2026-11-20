"""pFedSCPT: self-contained core implementation for personalized federated learning.

Run python main.py --help for options. See README.md for data and protocol details.
Extracted from the training implementation in experiments.py and FedConcat utilities.
"""

# MIT License
#
# Copyright (c) 2022 Anonymous authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import copy
import csv
import datetime
import json
import logging
import math
import os
import random
from functools import lru_cache
from math import ceil, log, sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

logger = logging.getLogger("pfedscpt")
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "tinyimagenet": 200}


# ==========================================================================
# 1. Configuration
# ==========================================================================

def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="pFedSCPT: selective class prototype transfer with persistent memory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    def option(group, name, value, help_text, choices=None):
        flags = ["--" + name.replace("_", "-")]
        if "_" in name:
            flags.append("--" + name)
        group.add_argument(*flags, dest=name, type=type(value), default=value,
                           help=help_text, choices=choices)

    run = parser.add_argument_group("Experiment")
    option(run, "dataset", "cifar100", "Dataset", list(NUM_CLASSES))
    option(run, "model", "auto", "auto selects SimpleCNN for CIFAR and ResNet-18 for Tiny-ImageNet",
           ["auto", "simple-cnn", "resnet18"])
    option(run, "datadir", "./data", "Dataset directory")
    option(run, "partition", "dirichlet", "homo, dirichlet (alias noniid), or noniid-#labelN")
    option(run, "beta", 0.1, "Dirichlet concentration alpha")
    option(run, "n_parties", 20, "Number of clients")
    option(run, "sample", 1.0, "Participation fraction after round 0")
    option(run, "comm_round", 200, "Communication rounds, including round-0 warm-up")
    run.add_argument("--seed", "--init-seed", "--init_seed", dest="init_seed", type=int,
                     default=0, help="Random seed")
    option(run, "device", "auto", "auto, cpu, or cuda:N")
    run.add_argument("--output-dir", type=Path, default=None,
                     help="New output directory; defaults to a timestamped runs/ directory")
    run.add_argument("--save-checkpoint", action="store_true",
                     help="Save final encoder, private adapters/heads, teachers, and memory")
    option(run, "num_threads", 4, "CPU threads used by PyTorch")

    opt = parser.add_argument_group("Local optimization")
    option(opt, "optimizer", "sgd", "Optimizer", ["sgd", "adam"])
    option(opt, "lr", -1.0, "Negative selects 0.005 for SimpleCNN or 0.1 for ResNet-18")
    option(opt, "batch_size", 100, "Local training batch size")
    option(opt, "epochs", 5, "Round-0 local CE warm-up epochs")
    option(opt, "e_distill", 7, "PRKI epochs per active round, following the reference launch script")
    option(opt, "rho", 0.9, "SGD momentum")
    option(opt, "reg", 1e-5, "Weight decay")
    option(opt, "ema_decay", 0.97, "Private teacher EMA decay")
    option(opt, "pure_local_warmup", 1, "1: CE-only round-0 warm-up", [0, 1])
    option(opt, "method_start_round", 1, "First active PRKI round (zero-based, at least 1)")
    option(opt, "method_ramp_rounds", 5, "Rounds to ramp method coefficients to full strength")
    option(opt, "method_ramp_min", 0.2, "Initial method coefficient scale")

    mem = parser.add_argument_group("CEMA class memory")
    option(mem, "use_cema_memory", 1, "Construct and use server class memory", [0, 1])
    option(mem, "local_qual_min_count", 7, "Minimum source class count n_0 before aggregation")
    option(mem, "class_reliable_min", 12, "Count at which local class reliability reaches one")
    option(mem, "global_gamma", 0.1, "Source class count exponent")
    option(mem, "global_min_support", 3, "Minimum supporting clients for strong memory")
    option(mem, "tau_B", 0.6, "Minimum reliability for strong memory")
    option(mem, "medium_min_support", 2, "Minimum supporting clients for medium memory")
    option(mem, "medium_tau_B", 0.3, "Minimum reliability for medium memory")
    option(mem, "medium_cema_max_weight", 0.5, "Maximum medium transfer weight")
    option(mem, "eta_B", 0.82, "Server prototype/variance EMA decay")

    transfer = parser.add_argument_group("PRKI and private knowledge preservation")
    for name, value, help_text in [
        ("lambda_loc", 0.30, "Private teacher preservation weight"),
        ("temp_local", 2.0, "Private teacher KL temperature"),
        ("gamma_hf", 0.30, "High-frequency feature preservation weight"),
        ("local_hf_ratio", 0.50, "High-frequency count / largest local class count"),
        ("lambda_fb", 0.05, "Gated shared prototype alignment weight"),
        ("gamma_global_proto", 1.0, "Gated alignment multiplier"),
        ("global_target_thresh", 0.20, "Minimum expert target weight for gated alignment"),
        ("lambda_personal_proto", 0.05, "Adapted RareP prototype alignment weight"),
        ("lambda_rare_ce", 0.20, "Extra CE weight for observed rare classes"),
        ("lambda_rep_seen", 0.20, "Observed rare class feature replay weight"),
        ("lambda_align", 0.05, "Shared prototype alignment weight"),
        ("lambda_pcon", 0.02, "Strong prototype contrastive weight"),
        ("gamma_proto_con", 0.05, "Positive value enables strong prototype contrastive computation"),
        ("proto_con_temp", 0.20, "Prototype contrastive temperature"),
        ("lambda_adapter_reg", 0.01, "Identity regularization weight for private adapters"),
        ("prior_adjust_tau", 0.10, "Training-only local label prior logit adjustment"),
        ("prior_eps", 1e-6, "Prior probability smoothing floor"),
        ("replay_per_class", 4, "Synthetic features per eligible class per real batch"),
        ("replay_noise", 0.05, "Scale multiplying diagonal replay standard deviation"),
        ("rare_replay_boost", 2.0, "Replay count multiplier for very small observed classes"),
        ("rare_replay_min_count", 8, "Boost replay below this positive local class count"),
        ("rare_head_ratio", 0.50, "Head class count / largest local class count"),
        ("rare_abs_threshold", 100, "Absolute observed rare class threshold"),
        ("rare_rel_threshold", 0.15, "Relative observed rare class threshold"),
    ]:
        option(transfer, name, value, help_text)
    args = parser.parse_args(argv)
    if args.model == "auto":
        args.model = "resnet18" if args.dataset == "tinyimagenet" else "simple-cnn"
    if args.lr < 0:
        args.lr = 0.1 if args.model == "resnet18" else 0.005
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.n_parties < 1 or not 0 < args.sample <= 1:
        parser.error("n-parties must be positive and sample must be in (0, 1]")
    for name in ["comm_round", "epochs", "e_distill", "batch_size", "num_threads",
                 "local_qual_min_count", "class_reliable_min", "method_ramp_rounds"]:
        if getattr(args, name) < 1:
            parser.error(name.replace("_", "-") + " must be positive")
    if args.lr <= 0 or args.reg < 0 or args.rho < 0:
        parser.error("lr must be positive; reg and rho must be nonnegative")
    if args.method_start_round < 1:
        parser.error("method-start-round must be at least 1; round 0 is warm-up")
    if args.beta <= 0 or not 0 <= args.method_ramp_min <= 1:
        parser.error("beta must be positive and method-ramp-min must be in [0, 1]")
    for name in ["ema_decay", "eta_B"]:
        if not 0 <= getattr(args, name) < 1:
            parser.error(name + " must be in [0, 1)")
    # Retained internal values: the standalone release uses clean image inputs
    # and explicit source count qualification, never the legacy reliability gate.
    args.noise = 0.0
    args.noise_type = "level"
    args.class_reliable_thresh = 0.55
    return args


# ==========================================================================
# 2. Shared encoder and private prediction modules
# ==========================================================================

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleEncoder, self).__init__()
        # if input_dim == 16*5*5:
        #     self.conv1 = nn.Conv2d(3, 6, 5)
        # else:
        #     self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.input_dim = input_dim

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.input_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class SimpleClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim=10):
        super(SimpleClassifier, self).__init__()

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = self.fc(x)
        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_header(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_header, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def ResNet18Encoder(**kwargs):
    r"""ResNet-18 encoder for split-model federated training."""
    return ResNet_header(BasicBlock, [2, 2, 2, 2], **kwargs)

class FeatureAffineAdapter(nn.Module):
    """Lightweight per-client feature adapter: A_k(u)=scale*u+bias."""

    def __init__(self, feature_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x):
        return x * self.scale.view(1, -1) + self.bias.view(1, -1)

    def identity_regularizer(self):
        return ((self.scale - 1.0) ** 2).mean() + (self.bias ** 2).mean()


# ==========================================================================
# 3. Dataset access and matched client train/test partition
# ==========================================================================

@lru_cache(maxsize=4)
def _cifar_source(root, train, dataset_name):
    cls = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    return cls(root=root, train=train, download=True)


class CIFARSubset(data.Dataset):
    """Reuse cached source arrays; preserve the original dataset transforms."""
    dataset_name = "cifar10"

    def __init__(self, root, dataidxs=None, train=True, transform=None, download=True):
        base = _cifar_source(os.path.abspath(root), bool(train), self.dataset_name)
        self.data = base.data if dataidxs is None else base.data[np.asarray(dataidxs, dtype=np.int64)]
        labels = np.asarray(base.targets, dtype=np.int64)
        self.target = labels if dataidxs is None else labels[np.asarray(dataidxs, dtype=np.int64)]
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        image = self.data[index]
        if self.dataset_name == "cifar100":
            image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.target[index])


class CIFAR10Subset(CIFARSubset):
    dataset_name = "cifar10"


class CIFAR100Subset(CIFARSubset):
    dataset_name = "cifar100"


def resolve_tinyimagenet_root(datadir):
    root = Path(datadir).expanduser().resolve()
    for candidate in [root / "tiny-imagenet-200", root]:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return str(candidate)
    raise FileNotFoundError("Extract Tiny-ImageNet first; expected train/ and val/ under " + str(root))


@lru_cache(maxsize=2)
def _tiny_samples(root):
    """Support both official flat validation images and class-folder validation."""
    root = Path(root)
    annotations = root / "val_annotations.txt"
    if annotations.is_file():
        train_folder = ImageFolder(str(root.parent / "train"))
        lookup = train_folder.class_to_idx
        samples = []
        for line in annotations.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 2:
                continue
            filename, wnid = fields[:2]
            if wnid not in lookup:
                raise ValueError("Unknown Tiny-ImageNet validation class: " + wnid)
            image_path = root / "images" / filename
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            samples.append((str(image_path), lookup[wnid]))
        # Same class/file order as torchvision ImageFolder after reorganization.
        return tuple(sorted(samples, key=lambda pair: (pair[1], pair[0])))
    folder = ImageFolder(str(root))
    if root.name == "val":
        train_folder = ImageFolder(str(root.parent / "train"))
        if folder.class_to_idx != train_folder.class_to_idx:
            raise ValueError("Tiny-ImageNet train/val class folders must have matching names")
    return tuple(folder.samples)


class ImageFolderSubset(data.Dataset):
    def __init__(self, root, dataidxs=None, transform=None):
        samples = _tiny_samples(os.path.abspath(root))
        indices = range(len(samples)) if dataidxs is None else dataidxs
        self.samples = [samples[int(i)] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, target = self.samples[index]
        with Image.open(filename) as image:
            image = image.convert("RGB")
        return self.transform(image) if self.transform else image, int(target)


def load_cifar10_data(datadir):
    train = _cifar_source(os.path.abspath(datadir), True, "cifar10")
    test = _cifar_source(os.path.abspath(datadir), False, "cifar10")
    return train.data, np.asarray(train.targets), test.data, np.asarray(test.targets)


def load_cifar100_data(datadir):
    train = _cifar_source(os.path.abspath(datadir), True, "cifar100")
    test = _cifar_source(os.path.abspath(datadir), False, "cifar100")
    return train.data, np.asarray(train.targets), test.data, np.asarray(test.targets)


def load_tinyimagenet_data(datadir):
    root = resolve_tinyimagenet_root(datadir)
    train = _tiny_samples(os.path.join(root, "train"))
    test = _tiny_samples(os.path.join(root, "val"))
    return (np.asarray([p for p, _ in train]), np.asarray([y for _, y in train]),
            np.asarray([p for p, _ in test]), np.asarray([y for _, y in test]))


class AddGaussianNoise:
    """Preserve the original CIFAR-10 transform's random-number consumption."""
    def __init__(self, mean=0., std=0., net_id=None, total=0):
        self.mean, self.std = mean, std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


def build_pfedscpt_paper_split(
    dataset,
    datadir,
    partition,
    n_parties,
    beta=0.5,
    seed=0,
    train_ratio=0.75,
):
    """
    pFedSCPT paper protocol: randomly split the whole dataset into train/test by 3:1,
    then assign both splits to clients with the same non-IID proportions so each
    client's local train and local test distributions are matched.
    """
    if dataset == 'cifar10':
        X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_tinyimagenet_data(datadir)
    else:
        raise NotImplementedError("pFedSCPT paper-style 3:1 split is currently implemented for cifar10/cifar100/tinyimagenet.")

    y_all = np.concatenate([np.array(y_train_raw), np.array(y_test_raw)]).astype(np.int64)
    raw_split_flags = np.concatenate([
        np.ones(len(y_train_raw), dtype=np.int64),
        np.zeros(len(y_test_raw), dtype=np.int64),
    ])
    raw_indices = np.concatenate([
        np.arange(len(y_train_raw), dtype=np.int64),
        np.arange(len(y_test_raw), dtype=np.int64),
    ])

    rng = np.random.RandomState(seed)
    all_positions = np.arange(len(y_all), dtype=np.int64)
    rng.shuffle(all_positions)
    split_point = int(len(all_positions) * train_ratio)
    split_point = min(max(split_point, 1), len(all_positions) - 1)
    train_pool = all_positions[:split_point].tolist()
    test_pool = all_positions[split_point:].tolist()

    partition_norm = str(partition).strip()
    if partition_norm in ("noniid", "noniid-", "labeldir", "dirichlet"):
        partition_norm = "noniid-labeldir"

    n_classes = int(len(np.unique(y_all)))
    proportions_by_class = {}
    if partition_norm == "homo":
        for cls_id in range(n_classes):
            proportions_by_class[cls_id] = np.repeat(1.0 / n_parties, n_parties)
    elif partition_norm == "noniid-labeldir":
        target_min_class_count = 0

        def _auto_anchor_floor(train_cls_count):
            """Internal safeguard for expert packages; not an experiment knob."""
            if train_cls_count <= 0:
                return 1
            if n_parties <= 1:
                return int(train_cls_count)
            feasible_two_client_floor = int(train_cls_count) // 2
            return max(1, min(target_min_class_count, feasible_two_client_floor))

        min_clients_per_class = 2 if n_parties >= 2 else 1
        auto_min_counts = []

        def _expert_friendly_proportions(raw_props, cls_id, min_class_count):
            """Keep label skew but avoid tiny class shards that cannot form reliable experts."""
            raw_props = np.asarray(raw_props, dtype=np.float64)
            raw_props = raw_props / max(raw_props.sum(), 1e-12)
            if min_class_count <= 0:
                return raw_props

            train_cls_count = int(np.sum(y_all[np.array(train_pool, dtype=np.int64)] == int(cls_id)))
            if train_cls_count < min_class_count:
                return raw_props

            max_active = max(1, min(n_parties, train_cls_count // min_class_count))
            expected_counts = raw_props * float(train_cls_count)
            dirichlet_active = int(np.sum(expected_counts >= float(min_class_count)))
            active_count = min(max_active, max(min_clients_per_class, dirichlet_active, 1))
            active_idx = np.argsort(raw_props)[-active_count:]

            floor_mass = min(1.0, float(active_count * min_class_count) / float(max(1, train_cls_count)))
            adjusted = np.zeros_like(raw_props)
            adjusted[active_idx] = float(min_class_count) / float(max(1, train_cls_count))

            remaining = max(0.0, 1.0 - floor_mass)
            active_raw = raw_props[active_idx]
            if remaining > 0.0 and active_raw.sum() > 0.0:
                adjusted[active_idx] += remaining * active_raw / active_raw.sum()

            if adjusted.sum() <= 0.0:
                adjusted = raw_props
            else:
                adjusted = adjusted / adjusted.sum()
            return adjusted

        for cls_id in range(n_classes):
            train_cls_count = int(np.sum(y_all[np.array(train_pool, dtype=np.int64)] == int(cls_id)))
            min_class_count = _auto_anchor_floor(train_cls_count)
            auto_min_counts.append(min_class_count)
            raw_props = rng.dirichlet(np.repeat(beta, n_parties))
            proportions_by_class[cls_id] = _expert_friendly_proportions(raw_props, cls_id, min_class_count)
        logger.info(
            "pFedSCPT expert-friendly Dirichlet | beta=%.4f | target_min_class_count=%d | auto_min_class_count_range=%d-%d | auto_min_clients_per_class=%d",
            beta,
            target_min_class_count,
            min(auto_min_counts) if auto_min_counts else 1,
            max(auto_min_counts) if auto_min_counts else 1,
            min_clients_per_class,
        )
    elif partition_norm.startswith("noniid-#label"):
        try:
            labels_per_client = int(partition_norm[len("noniid-#label"):])
        except ValueError:
            raise ValueError(
                f"Invalid partition format: {partition}. Expected 'noniid-#labelX', e.g. 'noniid-#label10'."
            )
        if labels_per_client <= 0:
            raise ValueError(f"labels_per_client must be positive, got {labels_per_client}.")
        if labels_per_client > n_classes:
            raise ValueError(
                f"labels_per_client={labels_per_client} exceeds number of classes={n_classes}."
            )
        if n_parties * labels_per_client < n_classes:
            raise ValueError(
                f"Cannot cover {n_classes} classes with {n_parties} clients and "
                f"{labels_per_client} labels per client. Increase --n_parties or --partition=noniid-#labelX."
            )

        client_labels = [[] for _ in range(n_parties)]
        class_owner_counts = np.zeros(n_classes, dtype=np.int64)
        for cls_id in rng.permutation(n_classes):
            eligible = [
                client_id for client_id in range(n_parties)
                if len(client_labels[client_id]) < labels_per_client
            ]
            if not eligible:
                raise RuntimeError("No eligible client left while assigning mandatory class coverage.")
            client_id = int(rng.choice(eligible))
            client_labels[client_id].append(int(cls_id))
            class_owner_counts[int(cls_id)] += 1

        for client_id in range(n_parties):
            while len(client_labels[client_id]) < labels_per_client:
                cls_id = int(rng.randint(0, n_classes))
                if cls_id in client_labels[client_id]:
                    continue
                client_labels[client_id].append(cls_id)
                class_owner_counts[cls_id] += 1

        for cls_id in range(n_classes):
            owners = [
                client_id for client_id, labels in enumerate(client_labels)
                if cls_id in labels
            ]
            if len(owners) == 0:
                raise RuntimeError(f"Class {cls_id} has no owner under {partition_norm}.")
            props = np.zeros(n_parties, dtype=np.float64)
            props[owners] = 1.0 / float(len(owners))
            proportions_by_class[cls_id] = props

        logger.info(
            "pFedSCPT fixed-label split | labels_per_client=%d | class_owner_count_range=%d-%d",
            labels_per_client,
            int(class_owner_counts.min()) if class_owner_counts.size > 0 else 0,
            int(class_owner_counts.max()) if class_owner_counts.size > 0 else 0,
        )
    else:
        raise NotImplementedError(
            "pFedSCPT paper-style split supports homo, Dirichlet noniid, and noniid-#labelX."
        )

    def _allocate_with_proportions(pool_positions):
        client_positions = [[] for _ in range(n_parties)]
        pool_positions = np.array(pool_positions, dtype=np.int64)
        for cls_id in np.unique(y_all):
            cls_positions = pool_positions[y_all[pool_positions] == cls_id].copy()
            rng.shuffle(cls_positions)
            if cls_positions.size == 0:
                continue
            proportions = proportions_by_class[int(cls_id)]
            proportions = np.asarray(proportions, dtype=np.float64)
            proportions = proportions / max(proportions.sum(), 1e-12)
            expected_counts = proportions * len(cls_positions)
            class_counts = np.floor(expected_counts + 1e-8).astype(int)
            class_counts[proportions <= 0.0] = 0
            remainder = int(len(cls_positions) - class_counts.sum())
            if remainder > 0:
                eligible = np.where(proportions > 0.0)[0]
                if eligible.size == 0:
                    eligible = np.arange(n_parties)
                fractional = expected_counts[eligible] - np.floor(expected_counts[eligible])
                order = eligible[np.argsort(-fractional)]
                for offset in range(remainder):
                    class_counts[int(order[offset % len(order)])] += 1
            elif remainder < 0:
                for client_id in np.argsort(-class_counts):
                    if remainder == 0:
                        break
                    removable = min(class_counts[client_id], -remainder)
                    class_counts[client_id] -= removable
                    remainder += removable
            split_points = np.cumsum(class_counts)[:-1]
            class_splits = np.split(cls_positions, split_points)
            client_positions = [idx_j + idx.tolist() for idx_j, idx in zip(client_positions, class_splits)]
        for client_id in range(n_parties):
            rng.shuffle(client_positions[client_id])
        return client_positions

    train_client_positions = _allocate_with_proportions(train_pool)
    test_client_positions = _allocate_with_proportions(test_pool)

    def _to_source_maps(client_positions):
        source_maps = {}
        for client_id, positions in enumerate(client_positions):
            positions = np.array(positions, dtype=np.int64)
            train_indices = raw_indices[positions[raw_split_flags[positions] == 1]]
            test_indices = raw_indices[positions[raw_split_flags[positions] == 0]]
            source_maps[client_id] = {
                'train': train_indices.astype(np.int64),
                'test': test_indices.astype(np.int64),
            }
        return source_maps

    train_counts = {}
    for client_id, positions in enumerate(train_client_positions):
        positions = np.array(positions, dtype=np.int64)
        unq, unq_cnt = np.unique(y_all[positions], return_counts=True) if positions.size > 0 else ([], [])
        train_counts[client_id] = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
    logger.info('pFedSCPT paper-style train data statistics: %s' % str(train_counts))

    return {
        'y_all': y_all,
        'train_pool': np.array(train_pool, dtype=np.int64),
        'test_pool': np.array(test_pool, dtype=np.int64),
        'train_position_map': {
            client_id: np.array(pos, dtype=np.int64)
            for client_id, pos in enumerate(train_client_positions)
        },
        'test_position_map': {
            client_id: np.array(pos, dtype=np.int64)
            for client_id, pos in enumerate(test_client_positions)
        },
        'train_source_map': _to_source_maps(train_client_positions),
        'test_source_map': _to_source_maps(test_client_positions),
        'train_counts': train_counts,
        'raw_split_flags': raw_split_flags.astype(np.int64),
        'raw_indices': raw_indices.astype(np.int64),
    }


# ==========================================================================
# 4. Fixed-shape CEMA statistics and reliability tiers
# ==========================================================================

def _resolve_cema_tier(
    support,
    reliability,
    strong_min_support,
    strong_tau,
    medium_min_support,
    medium_tau,
    medium_max_weight,
):
    support = int(support)
    reliability = float(min(max(reliability, 0.0), 1.0))
    if support >= int(strong_min_support) and reliability >= float(strong_tau):
        return 'strong', 1.0
    if support >= int(medium_min_support) and reliability >= float(medium_tau):
        return 'medium', min(reliability, float(medium_max_weight))
    return 'weak', 0.0

def _build_cema_fixed_upload(
    shared_prototypes,
    shared_variances,
    class_counts,
    class_reliabilities,
    num_classes,
    feature_dim,
    gamma,
    reliability_threshold,
    local_qual_min_count=0,
):
    """Build the fixed-shape CEMA contribution in the global class order.

    Each class contributes [a*rho*p, a*rho*var, a*rho, a, a*n, n].
    The optional quality factor q is one until a client-local validation score
    is available. This release uses q = 1.
    """
    num_classes = int(num_classes)
    feature_dim = int(feature_dim)
    class_width = 2 * feature_dim + 4
    upload = torch.zeros((num_classes, class_width), dtype=torch.float64)
    gamma = max(0.0, float(gamma))
    tau_r = float(min(max(reliability_threshold, 0.0), 1.0))
    local_qual_min_count = max(0, int(local_qual_min_count))

    for cls_id in range(num_classes):
        cls_count = max(0.0, float(class_counts.get(cls_id, 0.0)))
        upload[cls_id, 2 * feature_dim + 3] = cls_count
        if cls_count <= 0.0:
            continue

        reliability = float(min(max(class_reliabilities.get(cls_id, 0.0), 0.0), 1.0))
        proto = shared_prototypes.get(cls_id)
        # Source admission precedes continuous reliability weighting.
        # The CLI requires a positive sample-count threshold n_0.
        if local_qual_min_count > 0:
            eligible = cls_count >= float(local_qual_min_count) and proto is not None
        else:
            eligible = reliability >= tau_r and proto is not None
        if not eligible:
            continue

        proto = torch.nan_to_num(
            proto.detach().cpu().reshape(-1).to(dtype=torch.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if proto.numel() != feature_dim:
            raise ValueError(
                'CEMA prototype dimension mismatch for class %d: expected %d, got %d'
                % (cls_id, feature_dim, proto.numel())
            )
        proto = F.normalize(proto, dim=0)

        variance = shared_variances.get(cls_id)
        if variance is None:
            variance = torch.zeros_like(proto)
        else:
            variance = torch.nan_to_num(
                variance.detach().cpu().reshape(-1).to(dtype=torch.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if variance.numel() != feature_dim:
                raise ValueError(
                    'CEMA variance dimension mismatch for class %d: expected %d, got %d'
                    % (cls_id, feature_dim, variance.numel())
                )
            variance = torch.clamp(variance, min=0.0)

        quality = 1.0
        rho = (cls_count ** gamma) * reliability * quality
        upload[cls_id, :feature_dim] = rho * proto
        upload[cls_id, feature_dim:2 * feature_dim] = rho * variance
        upload[cls_id, 2 * feature_dim] = rho
        upload[cls_id, 2 * feature_dim + 1] = 1.0
        upload[cls_id, 2 * feature_dim + 2] = cls_count

    return upload.reshape(-1)

def _accumulate_cema_secure_sum(aggregate, client_upload):
    """Simulate the aggregate-only output of SecAgg without storing packets."""
    client_upload = client_upload.detach().cpu().to(dtype=torch.float64)
    if aggregate is None:
        return client_upload.clone()
    if aggregate.shape != client_upload.shape:
        raise ValueError(
            'CEMA fixed-upload shape mismatch: aggregate=%s client=%s'
            % (tuple(aggregate.shape), tuple(client_upload.shape))
        )
    aggregate.add_(client_upload)
    return aggregate

def _view_cema_secure_sum(secure_sum, num_classes, feature_dim):
    class_width = 2 * int(feature_dim) + 4
    expected = int(num_classes) * class_width
    if secure_sum is None:
        return None
    secure_sum = secure_sum.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if secure_sum.numel() != expected:
        raise ValueError(
            'CEMA secure-sum dimension mismatch: expected %d, got %d'
            % (expected, secure_sum.numel())
        )
    return secure_sum.reshape(int(num_classes), class_width)


# ==========================================================================
# 5. Federated training: CEMA, private teachers, and PRKI
# ==========================================================================

def run_federated(args, split):
    """Run the inherited round schedule; return final client models and class memory.

    Round 0 is private CE warm-up. Later rounds aggregate/replace the encoder,
    collect CEMA statistics, update server memory, refresh private teachers,
    and run PRKI. Adapters, classifiers and BN buffers are never averaged.
    """
    device = torch.device(args.device)
    y_train = split["y_all"]
    pfedscpt_paper_split = split
    history = []
    client_metrics = []
    logger.info("Initializing pFedSCPT nets")

    if not (1 <= int(args.medium_min_support) < int(args.global_min_support)):
        raise ValueError(
            "Tiered CEMA requires 1 <= medium_min_support < global_min_support."
        )
    if not (0.0 <= float(args.medium_tau_B) < float(args.tau_B) <= 1.0):
        raise ValueError("Tiered CEMA requires 0 <= medium_tau_B < tau_B <= 1.")
    if not (0.0 < float(args.medium_cema_max_weight) <= 0.5):
        raise ValueError("Tiered CEMA requires 0 < medium_cema_max_weight <= 0.5.")
    if int(getattr(args, 'adaptive_tau_B', 0)) != 0:
        logger.warning(
            "adaptive_tau_B=%d is ignored by the fixed-threshold tiered CEMA policy.",
            int(args.adaptive_tau_B),
        )

    num_ftrs = 512 if args.model == "resnet18" else 256
    output_dim = NUM_CLASSES[args.dataset]

    encoder_list = []
    adapter_list = []
    classifier_list = []
    for i in range(args.n_parties):
        encoder = (ResNet18Encoder() if args.model == "resnet18" else
                   SimpleEncoder(input_dim=4096, hidden_dims=[512, 256], output_dim=10))
        encoder_list.append(encoder)
        classifier_list.append(SimpleClassifier(hidden_dim=num_ftrs, output_dim=output_dim))
        adapter_list.append(FeatureAffineAdapter(num_ftrs))

    global_encoder_init = copy.deepcopy(encoder_list[0])
    classifier_init = copy.deepcopy(classifier_list[0])
    for enc in encoder_list:
        enc.load_state_dict(global_encoder_init.state_dict())
    for cls in classifier_list:
        cls.load_state_dict(classifier_init.state_dict())

    private_anchor_encoders = {k: None for k in range(args.n_parties)}
    private_anchor_adapters = {k: None for k in range(args.n_parties)}
    private_anchor_classifiers = {k: None for k in range(args.n_parties)}
    client_proto_map = {k: {} for k in range(args.n_parties)}
    class_expert_memory = {}
    class_expert_tiers = {}
    client_trainidx_map = {}
    client_testidx_map = {}
    client_train_source_map = {}
    client_test_source_map = {}

    def _count_labels(dataidxs):
        counts = {c: 0 for c in range(output_dim)}
        local_targets = np.array(y_train[np.array(dataidxs, dtype=np.int64)]) if len(dataidxs) > 0 else np.array([])
        if local_targets.size > 0:
            unique_classes, class_counts = np.unique(local_targets, return_counts=True)
            for cls_id, cls_cnt in zip(unique_classes, class_counts):
                counts[int(cls_id)] = int(cls_cnt)
        return counts

    for k in range(args.n_parties):
        if pfedscpt_paper_split is None:
            raise RuntimeError("pFedSCPT requires the paper-style split metadata built before entering its branch.")
        train_idx_k = pfedscpt_paper_split['train_position_map'][k]
        test_idx_k = pfedscpt_paper_split['test_position_map'][k]
        client_trainidx_map[k] = train_idx_k
        client_testidx_map[k] = test_idx_k
        client_train_source_map[k] = pfedscpt_paper_split['train_source_map'][k]
        client_test_source_map[k] = pfedscpt_paper_split['test_source_map'][k]

    def _make_pfedscpt_source_dataset(source_map, is_train, noise_level=0, net_id=None):
        if args.dataset == 'cifar10':
            dl_obj = CIFAR10Subset
            if is_train:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze()),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, args.n_parties - 1),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    AddGaussianNoise(0., noise_level, net_id, args.n_parties - 1),
                ])
        elif args.dataset == 'cifar100':
            dl_obj = CIFAR100Subset
            normalize = transforms.Normalize(
                mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
            )
            if is_train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                transform = transforms.Compose([transforms.ToTensor(), normalize])
        elif args.dataset == 'tinyimagenet':
            dl_obj = ImageFolderSubset
            tinyimagenet_root = resolve_tinyimagenet_root(args.datadir)
            train_root = os.path.join(tinyimagenet_root, 'train')
            val_root = os.path.join(tinyimagenet_root, 'val')
            if is_train:
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        else:
            raise NotImplementedError("pFedSCPT paper source dataloader is implemented for cifar10/cifar100/tinyimagenet.")

        datasets_local = []
        if args.dataset == 'tinyimagenet':
            if len(source_map.get('train', [])) > 0:
                datasets_local.append(dl_obj(train_root, dataidxs=source_map['train'], transform=transform))
            if len(source_map.get('test', [])) > 0:
                datasets_local.append(dl_obj(val_root, dataidxs=source_map['test'], transform=transform))
            if len(datasets_local) == 0:
                raise ValueError("Empty client split; choose another seed, partition, or client count")
        else:
            if len(source_map.get('train', [])) > 0:
                datasets_local.append(dl_obj(args.datadir, dataidxs=source_map['train'], train=True, transform=transform, download=True))
            if len(source_map.get('test', [])) > 0:
                datasets_local.append(dl_obj(args.datadir, dataidxs=source_map['test'], train=False, transform=transform, download=True))
            if len(datasets_local) == 0:
                raise ValueError("Empty client split; choose another seed, partition, or client count")
        if len(datasets_local) == 1:
            return datasets_local[0]
        return data.ConcatDataset(datasets_local)

    def _get_pfedscpt_paper_loader(client_id, is_train, batch_size, noise_level=0):
        source_map = client_train_source_map[client_id] if is_train else client_test_source_map[client_id]
        dataset_local = _make_pfedscpt_source_dataset(source_map, is_train=is_train, noise_level=noise_level, net_id=client_id)
        return data.DataLoader(dataset=dataset_local, batch_size=batch_size, shuffle=is_train, drop_last=False)

    def _train_pfedscpt_warmup(selected_clients):
        pure_local_warmup = int(getattr(args, 'pure_local_warmup', 1)) == 1
        for net_id in selected_clients:
            train_dl_local = _get_pfedscpt_paper_loader(net_id, is_train=True, batch_size=args.batch_size)
            logger.info("Training network %s. n_training: %d" % (str(net_id), len(train_dl_local.dataset)))
            encoder_list[net_id].to(device).train()
            adapter_list[net_id].to(device).train()
            classifier_list[net_id].to(device).train()
            warmup_params = (
                list(encoder_list[net_id].parameters()) +
                list(adapter_list[net_id].parameters()) +
                list(classifier_list[net_id].parameters())
            )
            if args.optimizer == 'adam':
                optimizer = optim.Adam(
                    warmup_params,
                    lr=args.lr,
                    weight_decay=args.reg,
                )
            else:
                optimizer = optim.SGD(
                    warmup_params,
                    lr=args.lr,
                    momentum=args.rho,
                    weight_decay=args.reg,
                )
            for epoch in range(args.epochs):
                loss_collector = []
                for x, target in train_dl_local:
                    x = x.to(device)
                    target = target.to(device, dtype=torch.int64)
                    optimizer.zero_grad()
                    feat_shared = encoder_list[net_id](x)
                    feat_personal = adapter_list[net_id](feat_shared)
                    logits_personal = classifier_list[net_id](feat_personal)
                    loss = F.cross_entropy(logits_personal, target)
                    if not pure_local_warmup:
                        loss = (
                            loss +
                            args.lambda_adapter_reg * adapter_list[net_id].identity_regularizer()
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(warmup_params, max_norm=2.0)
                    optimizer.step()
                    loss_collector.append(loss.item())
                if len(loss_collector) > 0:
                    logger.info('Client %d Warmup Epoch %d Loss: %.6f' % (net_id, epoch, float(np.mean(loss_collector))))
            encoder_list[net_id].to("cpu")
            adapter_list[net_id].to("cpu")
            classifier_list[net_id].to("cpu")

    def _compute_local_prototypes(client_id, encoder_model, adapter_model=None):
        noise_level = args.noise if client_id != args.n_parties - 1 else 0
        train_dl_local = _get_pfedscpt_paper_loader(client_id, is_train=True, batch_size=args.batch_size, noise_level=noise_level)
        encoder_model = encoder_model.to(device)
        adapter_model = adapter_model.to(device) if adapter_model is not None else None
        encoder_model.eval()
        if adapter_model is not None:
            adapter_model.eval()
        shared_sums = {}
        shared_sq_sums = {}
        proto_sums = {}
        proto_counts = {}
        with torch.no_grad():
            for x, target in train_dl_local:
                x = x.to(device)
                target = target.to(device)
                feat_shared = F.normalize(encoder_model(x), dim=1)
                feat = adapter_model(feat_shared) if adapter_model is not None else feat_shared
                feat = F.normalize(feat, dim=1)
                for cls_id in torch.unique(target):
                    cls_int = int(cls_id.item())
                    cls_mask = target == cls_id
                    cls_feat = feat[cls_mask]
                    cls_shared = feat_shared[cls_mask]
                    if cls_feat.numel() == 0:
                        continue
                    feat_sum = cls_feat.sum(dim=0)
                    shared_sum = cls_shared.sum(dim=0)
                    shared_sq_sum = (cls_shared ** 2).sum(dim=0)
                    if cls_int not in proto_sums:
                        proto_sums[cls_int] = feat_sum.detach().clone()
                        shared_sums[cls_int] = shared_sum.detach().clone()
                        shared_sq_sums[cls_int] = shared_sq_sum.detach().clone()
                        proto_counts[cls_int] = int(cls_feat.shape[0])
                    else:
                        proto_sums[cls_int] += feat_sum.detach()
                        shared_sums[cls_int] += shared_sum.detach()
                        shared_sq_sums[cls_int] += shared_sq_sum.detach()
                        proto_counts[cls_int] += int(cls_feat.shape[0])

        prototypes = {}
        shared_prototypes = {}
        shared_vars = {}
        for cls_int, feat_sum in proto_sums.items():
            prototypes[cls_int] = F.normalize(
                feat_sum / max(1, proto_counts[cls_int]),
                dim=0,
            ).detach().cpu()
            shared_mean = shared_sums[cls_int] / max(1, proto_counts[cls_int])
            shared_prototypes[cls_int] = F.normalize(shared_mean, dim=0).detach().cpu()
            shared_vars[cls_int] = torch.clamp(
                shared_sq_sums[cls_int] / max(1, proto_counts[cls_int]) - shared_mean ** 2,
                min=0.0,
            ).detach().cpu()
        encoder_model.to("cpu")
        if adapter_model is not None:
            adapter_model.to("cpu")
        return prototypes, shared_prototypes, shared_vars

    def _class_reliability(client_id, cls_id):
        class_counts = _count_labels(client_trainidx_map[client_id])
        min_count = max(1, int(getattr(args, 'class_reliable_min', 8)))
        return min(1.0, float(class_counts.get(int(cls_id), 0)) / float(min_count))

    def _client_class_groups(client_id):
        class_counts = _count_labels(client_trainidx_map[client_id])
        positive_counts = [int(cnt) for cnt in class_counts.values() if int(cnt) > 0]
        max_count = max(positive_counts) if len(positive_counts) > 0 else 0
        tau_head = float(min(max(getattr(args, 'rare_head_ratio', 0.5), 0.0), 1.0))
        tau_abs = max(1, int(getattr(args, 'rare_abs_threshold', 100)))
        tau_rel = float(max(getattr(args, 'rare_rel_threshold', 0.15), 0.0))
        rare_limit = max(float(tau_abs), float(tau_rel) * float(max_count))
        head_limit = float(tau_head) * float(max_count)
        present = {
            int(cls_id)
            for cls_id, cls_cnt in class_counts.items()
            if int(cls_cnt) > 0
        }
        head = {
            int(cls_id)
            for cls_id, cls_cnt in class_counts.items()
            if int(cls_cnt) > 0 and float(cls_cnt) >= head_limit
        }
        rare_present = {
            int(cls_id)
            for cls_id, cls_cnt in class_counts.items()
            if int(cls_cnt) > 0 and float(cls_cnt) <= rare_limit
        }
        missing = {
            int(cls_id)
            for cls_id in range(output_dim)
            if int(class_counts.get(int(cls_id), 0)) == 0
        }
        return {
            'present': present,
            'head': head,
            'rare_present': rare_present,
            'missing': missing,
            'max_count': max_count,
            'rare_limit': rare_limit,
            'head_limit': head_limit,
        }

    def _format_count_acc(correct, total):
        if int(total) <= 0:
            return "N/A(0/0)"
        return "%.4f(%d/%d)" % (float(correct) / float(total), int(correct), int(total))

    def _merge_prototype_memory(memory_dict, current_proto, ema_decay):
        merged = {}
        ema_decay = float(min(max(ema_decay, 0.0), 0.9999))
        memory_proto = memory_dict if memory_dict is not None else {}
        class_ids = set(memory_proto.keys()) | set(current_proto.keys())
        for cls_id in class_ids:
            cur = current_proto.get(cls_id)
            old = memory_proto.get(cls_id)
            if cur is None and old is None:
                continue
            if cur is None:
                merged[cls_id] = old.clone()
                continue
            cur = torch.nan_to_num(cur.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
            cur = F.normalize(cur, dim=0)
            if old is None:
                merged[cls_id] = cur
            else:
                old = torch.nan_to_num(old.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
                old = F.normalize(old, dim=0)
                merged_vec = ema_decay * old + (1.0 - ema_decay) * cur
                merged[cls_id] = F.normalize(merged_vec, dim=0)
        return merged

    def _refresh_client_statistics(client_id, build_cema_upload=False):
        personal_proto, shared_proto, shared_var = _compute_local_prototypes(
            client_id,
            encoder_list[client_id],
            adapter_list[client_id],
        )
        # Personalized prototypes remain client-local. Shared-space CEMA
        # statistics are returned only as one transient fixed-shape packet.
        client_proto_map[client_id] = personal_proto
        if not build_cema_upload:
            return None

        class_counts = _count_labels(client_trainidx_map[client_id])
        min_count = max(1, int(getattr(args, 'class_reliable_min', 8)))
        class_reliabilities = {
            int(cls_id): min(1.0, float(cls_count) / float(min_count))
            for cls_id, cls_count in class_counts.items()
        }
        return _build_cema_fixed_upload(
            shared_prototypes=shared_proto,
            shared_variances=shared_var,
            class_counts=class_counts,
            class_reliabilities=class_reliabilities,
            num_classes=output_dim,
            feature_dim=num_ftrs,
            gamma=getattr(args, 'global_gamma', 0.3),
            reliability_threshold=getattr(args, 'class_reliable_thresh', 0.5),
            local_qual_min_count=getattr(args, 'local_qual_min_count', 0),
        )

    def _class_expert_tier(info):
        if info is None:
            return 'weak', 0.0
        return _resolve_cema_tier(
            support=info.get('support', 0),
            reliability=info.get('reliability', 0.0),
            strong_min_support=getattr(args, 'global_min_support', 3),
            strong_tau=getattr(args, 'tau_B', 0.60),
            medium_min_support=getattr(args, 'medium_min_support', 2),
            medium_tau=getattr(args, 'medium_tau_B', 0.30),
            medium_max_weight=getattr(args, 'medium_cema_max_weight', 0.50),
        )

    def _class_tier_decision(cls_id):
        return class_expert_tiers.get(int(cls_id), ('weak', 0.0))

    def _build_class_expert_memory(secure_aggregate):
        current_memory = {}
        min_support = max(1, int(getattr(args, 'global_min_support', 3)))
        secure_view = _view_cema_secure_sum(
            secure_aggregate,
            num_classes=output_dim,
            feature_dim=num_ftrs,
        )
        if secure_view is not None:
            for cls_id in range(output_dim):
                row = secure_view[cls_id]
                proto_sum = row[:num_ftrs]
                var_sum = row[num_ftrs:2 * num_ftrs]
                denom = float(row[2 * num_ftrs].item())
                if denom <= 0.0:
                    continue

                support = int(np.rint(float(row[2 * num_ftrs + 1].item())))
                eligible_cls_count = float(row[2 * num_ftrs + 2].item())
                total_cls_count = float(row[2 * num_ftrs + 3].item())
                support_score = min(1.0, float(support) / float(min_support))
                coverage_score = min(1.0, eligible_cls_count / max(total_cls_count, 1.0))
                reliability = support_score * coverage_score
                proto_center = F.normalize(proto_sum / denom, dim=0)
                # For unit-normalized client prototypes, ||sum rho*p||/sum rho
                # exactly equals their weighted cosine agreement with the center.
                prototype_agreement = min(
                    max(float(torch.linalg.vector_norm(proto_sum).item()) / denom, 0.0),
                    1.0,
                )
                current_memory[int(cls_id)] = {
                    'proto': proto_center.to(dtype=torch.float32).detach().cpu(),
                    'var': torch.clamp(var_sum / denom, min=0.0).to(dtype=torch.float32).detach().cpu(),
                    'support': support,
                    'reliability': float(min(max(reliability, 0.0), 1.0)),
                    'denom': float(denom),
                    'eligible_count': float(eligible_cls_count),
                    'total_count': float(total_cls_count),
                    'prototype_agreement': float(prototype_agreement),
                }

        eta_b = float(min(max(getattr(args, 'eta_B', 0.8), 0.0), 0.9999))
        merged = {}
        for cls_id in set(class_expert_memory.keys()) | set(current_memory.keys()):
            cur = current_memory.get(cls_id)
            old = class_expert_memory.get(cls_id)
            if cur is None:
                merged[cls_id] = old
                continue
            if old is None:
                merged[cls_id] = cur
                continue
            merged[cls_id] = {
                'proto': F.normalize(eta_b * old['proto'] + (1.0 - eta_b) * cur['proto'], dim=0).detach().cpu(),
                'var': torch.clamp(eta_b * old['var'] + (1.0 - eta_b) * cur['var'], min=0.0).detach().cpu(),
                'support': max(int(old.get('support', 0)), int(cur.get('support', 0))),
                'reliability': max(float(old.get('reliability', 0.0)), float(cur.get('reliability', 0.0))),
                'denom': max(float(old.get('denom', 0.0)), float(cur.get('denom', 0.0))),
                'eligible_count': max(float(old.get('eligible_count', 0.0)), float(cur.get('eligible_count', 0.0))),
                'total_count': max(float(old.get('total_count', 0.0)), float(cur.get('total_count', 0.0))),
                'prototype_agreement': (
                    eta_b * float(old.get('prototype_agreement', cur.get('prototype_agreement', 0.0))) +
                    (1.0 - eta_b) * float(cur.get('prototype_agreement', 0.0))
                ),
            }
        return merged

    def _ema_update_module_state(anchor_module, student_module, ema_decay):
        anchor_state = anchor_module.state_dict()
        student_state = student_module.state_dict()
        for key, anchor_tensor in anchor_state.items():
            student_tensor = student_state[key].to(device=anchor_tensor.device)
            if anchor_tensor.is_floating_point():
                anchor_tensor.mul_(ema_decay).add_(
                    student_tensor.to(dtype=anchor_tensor.dtype),
                    alpha=1.0 - ema_decay,
                )
            else:
                anchor_tensor.copy_(student_tensor.to(dtype=anchor_tensor.dtype))
        anchor_module.load_state_dict(anchor_state)

    def _pfedscpt_method_scale(round_idx):
        start_round = max(0, int(getattr(args, 'method_start_round', 1)))
        if round_idx < start_round:
            return 0.0
        ramp_rounds = max(1, int(getattr(args, 'method_ramp_rounds', 5)))
        ramp_min = float(min(max(getattr(args, 'method_ramp_min', 0.2), 0.0), 1.0))
        if ramp_rounds == 1:
            return 1.0
        progress = min(max(float(round_idx - start_round) / float(ramp_rounds - 1), 0.0), 1.0)
        return ramp_min + (1.0 - ramp_min) * progress

    # Aggregate only the shared encoder with client-local BN statistics.
    def _fedavg_module_list(module_list, selected_ids, tag):
        def _is_bn_buffer(key):
            return ('running_mean' in key) or ('running_var' in key) or ('num_batches_tracked' in key)
        with torch.no_grad():
            sizes = {cid: max(1, len(client_trainidx_map[cid])) for cid in selected_ids}
            total = float(sum(sizes.values()))
            ref_state = module_list[selected_ids[0]].state_dict()
            avg_state = {
                key: torch.zeros_like(val, dtype=torch.float32)
                for key, val in ref_state.items() if not _is_bn_buffer(key)
            }
            for cid in selected_ids:
                w = sizes[cid] / total
                st = module_list[cid].state_dict()
                for key in avg_state:
                    avg_state[key] += st[key].float().to(avg_state[key].device) * w
            for cid in range(args.n_parties):
                st = module_list[cid].state_dict()
                for key in avg_state:
                    st[key].copy_(avg_state[key].to(dtype=st[key].dtype, device=st[key].device))
                module_list[cid].load_state_dict(st)
        logger.info(
            "pFedSCPT aggregation | module=%s | selected=%d | total_samples=%d | BN buffers kept local",
            tag, len(selected_ids), int(total),
        )

    for round in range(args.comm_round):
        current_lr = args.lr * (0.5 ** (round // 40))
        logger.info("========== pFedSCPT Comm Round: %d ==========" % round)
        round_cema_secure_sum = None

        if round == 0:
            selected = list(range(args.n_parties))
        else:
            top = max(1, int(args.n_parties * args.sample))
            selected = np.random.permutation(np.arange(args.n_parties))[:top].tolist()

        # Round 0: local CE warm-up and initialization of private frozen teachers.
        if round == 0:
            warmup_epochs_backup = args.epochs
            warmup_lr_backup = args.lr
            _train_pfedscpt_warmup(selected)
            args.epochs = warmup_epochs_backup
            args.lr = warmup_lr_backup
            # Round 0 keeps each client's personalized warm-up state local.
            for k in selected:
                private_anchor_encoders[k] = copy.deepcopy(encoder_list[k]).to("cpu")
                private_anchor_adapters[k] = copy.deepcopy(adapter_list[k]).to("cpu")
                private_anchor_classifiers[k] = copy.deepcopy(classifier_list[k]).to("cpu")
                _refresh_client_statistics(k)
                private_anchor_encoders[k].eval()
                private_anchor_adapters[k].eval()
                private_anchor_classifiers[k].eval()
                for p_ in private_anchor_encoders[k].parameters():
                    p_.requires_grad = False
                for p_ in private_anchor_adapters[k].parameters():
                    p_.requires_grad = False
                for p_ in private_anchor_classifiers[k].parameters():
                    p_.requires_grad = False
        else:
            logger.info("Skip warm-up for round %d and refresh client statistics for the next stage." % round)
            # Aggregate only the shared encoder; adapters and personalized heads remain local.
            _fedavg_module_list(encoder_list, selected, "shared_encoder(E_g)")
            collect_cema_upload = (
                _pfedscpt_method_scale(round) > 0.0
                and int(getattr(args, 'use_cema_memory', 1)) != 0
            )
            for k in selected:
                client_cema_upload = _refresh_client_statistics(
                    k,
                    build_cema_upload=collect_cema_upload,
                )
                if client_cema_upload is not None:
                    round_cema_secure_sum = _accumulate_cema_secure_sum(
                        round_cema_secure_sum,
                        client_cema_upload,
                    )
            if round_cema_secure_sum is not None:
                logger.info(
                    "CEMA upload | clients=%d | fixed_dim=%d | server_visible=class_sums_only | secagg_backend=sum_simulation_no_crypto",
                    len(selected),
                    int(round_cema_secure_sum.numel()),
                )

        method_round_scale = _pfedscpt_method_scale(round)
        use_cema_memory = int(getattr(args, 'use_cema_memory', 1)) != 0
        logger.info(
            "pFedSCPT method schedule | round=%d | scale=%.4f | pure_local_warmup=%d | start_round=%d | ramp_rounds=%d | use_cema_memory=%d",
            round,
            method_round_scale,
            int(getattr(args, 'pure_local_warmup', 1)),
            int(getattr(args, 'method_start_round', 1)),
            int(getattr(args, 'method_ramp_rounds', 5)),
            int(use_cema_memory),
        )

        if method_round_scale <= 0.0:
            logger.info("Method inactive for round %d; skip class-expert memory/replay and keep pure local training only.", round)
            class_expert_memory = {}
            class_expert_tiers = {}
        elif not use_cema_memory:
            class_expert_memory = {}
            class_expert_tiers = {}
            logger.info("CEMA memory disabled for round %d; skip class-expert memory construction.", round)
        else:
            class_expert_memory = _build_class_expert_memory(round_cema_secure_sum)
            class_expert_tiers = {
                int(cls_id): _class_expert_tier(info)
                for cls_id, info in class_expert_memory.items()
            }
            tier_counts = {'strong': 0, 'medium': 0, 'weak': 0}
            for tier, _ in class_expert_tiers.values():
                tier_counts[tier] += 1
            logger.info(
                "CEMA tier memory | strong_classes=%d | medium_classes=%d | weak_classes=%d | candidate_classes=%d",
                tier_counts['strong'],
                tier_counts['medium'],
                tier_counts['weak'],
                len(class_expert_memory),
            )


        # PRKI: real-image training and class feature replay using current memory.
        for k in selected:
            if method_round_scale <= 0.0:
                continue
            private_anchor_encoders[k].to(device)
            private_anchor_adapters[k].to(device)
            private_anchor_classifiers[k].to(device)
            # Refresh the private teacher by EMA before local PRKI updates.
            if round > 0:
                with torch.no_grad():
                    ema_decay = min(max(args.ema_decay, 0.0), 0.9999)
                    _ema_update_module_state(private_anchor_encoders[k], encoder_list[k], ema_decay)
                    _ema_update_module_state(private_anchor_adapters[k], adapter_list[k], ema_decay)
                    _ema_update_module_state(private_anchor_classifiers[k], classifier_list[k], ema_decay)

            for m_ in [private_anchor_encoders[k], private_anchor_adapters[k], private_anchor_classifiers[k]]:
                m_.eval()
                for p_ in m_.parameters():
                    p_.requires_grad = False

            noise_level = args.noise if k != args.n_parties - 1 else 0
            train_dl_local = _get_pfedscpt_paper_loader(k, is_train=True, batch_size=args.batch_size, noise_level=noise_level)

            student_encoder = encoder_list[k].to(device)
            student_adapter = adapter_list[k].to(device)
            student_classifier = classifier_list[k].to(device)
            student_encoder.train()
            student_adapter.train()
            student_classifier.train()

            if args.optimizer == 'adam':
                optimizer = optim.Adam(
                    list(student_encoder.parameters()) +
                    list(student_adapter.parameters()) +
                    list(student_classifier.parameters()),
                    lr=current_lr,
                    weight_decay=args.reg,
                )
            else:
                optimizer = optim.SGD(
                    list(student_encoder.parameters()) +
                    list(student_adapter.parameters()) +
                    list(student_classifier.parameters()),
                    lr=current_lr,
                    momentum=args.rho,
                    weight_decay=args.reg,
                )

            ce_loss = nn.CrossEntropyLoss().to(device)
            local_temperature = max(float(getattr(args, 'temp_local', 2.0)), 1e-6)
            global_target_thresh = float(min(max(getattr(args, 'global_target_thresh', 0.20), 0.0), 1.0))
            local_hf_ratio = float(min(max(getattr(args, 'local_hf_ratio', 0.5), 0.0), 1.0))
            gamma_hf = float(max(getattr(args, 'gamma_hf', 0.5), 0.0))
            gamma_proto_con = float(max(getattr(args, 'gamma_proto_con', 0.2), 0.0))
            gamma_global_proto = float(max(getattr(args, 'gamma_global_proto', 1.0), 0.0))
            lambda_fb = float(max(getattr(args, 'lambda_fb', 0.12), 0.0))
            prior_adjust_tau = method_round_scale * float(max(getattr(args, 'prior_adjust_tau', 0.2), 0.0))
            prior_eps = float(max(getattr(args, 'prior_eps', 1e-6), 1e-12))
            proto_con_temp = max(float(getattr(args, 'proto_con_temp', 0.2)), 1e-6)
            label_count_map = {}
            local_targets_np = np.array(y_train[client_trainidx_map[k]])
            if local_targets_np.size > 0:
                unique_classes, class_counts = np.unique(local_targets_np, return_counts=True)
                for cls_id, cls_cnt in zip(unique_classes, class_counts):
                    label_count_map[int(cls_id)] = int(cls_cnt)
                max_class_count = float(np.max(class_counts))
            else:
                max_class_count = 1.0
            class_groups = _client_class_groups(k)
            rare_present_class_set = class_groups['rare_present']
            present_total = max(1.0, float(sum(label_count_map.values())))
            prior_log_vec = torch.zeros(output_dim, dtype=torch.float32, device=device)
            for cls_id in range(output_dim):
                cls_count = float(label_count_map.get(cls_id, 0))
                if cls_count > 0.0:
                    prior_prob = max(cls_count / present_total, prior_eps)
                    prior_log_vec[cls_id] = log(prior_prob)

            def _apply_prior_adjustment(logits_tensor):
                if prior_adjust_tau <= 0.0:
                    return logits_tensor
                return logits_tensor - prior_log_vec.to(
                    device=logits_tensor.device,
                    dtype=logits_tensor.dtype,
                ).unsqueeze(0) * prior_adjust_tau

            def _prototype_contrastive_losses(
                feat_norm_tensor,
                target_tensor,
                proto_matrix,
                proto_index_lookup,
            ):
                losses = torch.zeros(feat_norm_tensor.shape[0], dtype=feat_norm_tensor.dtype, device=feat_norm_tensor.device)
                if gamma_proto_con <= 0.0 or proto_matrix is None or proto_matrix.shape[0] <= 1:
                    return losses
                proto_logits = torch.matmul(
                    feat_norm_tensor,
                    proto_matrix.to(
                        device=feat_norm_tensor.device,
                        dtype=feat_norm_tensor.dtype,
                    ).t(),
                ) / proto_con_temp
                proto_targets = proto_index_lookup.index_select(0, target_tensor)
                valid_targets = proto_targets.ge(0)
                safe_targets = proto_targets.clamp_min(0)
                all_losses = F.cross_entropy(
                    proto_logits,
                    safe_targets,
                    reduction='none',
                )
                return all_losses * valid_targets.to(dtype=all_losses.dtype)

            strong_expert_classes = [
                int(cls_id)
                for cls_id in class_expert_memory
                if _class_tier_decision(cls_id)[0] == 'strong'
            ]
            def _sample_expert_features(class_ids, per_class):
                feats = []
                labels = []
                class_weights = []
                if not isinstance(per_class, dict) and per_class <= 0:
                    return None, None, None
                for cls_id in class_ids:
                    if isinstance(per_class, dict):
                        cls_per_class = int(per_class.get(int(cls_id), 0))
                    else:
                        cls_per_class = int(per_class)
                    if cls_per_class <= 0:
                        continue
                    info = class_expert_memory.get(int(cls_id))
                    if info is None:
                        continue
                    _, cema_weight = _class_tier_decision(cls_id)
                    if cema_weight <= 0.0:
                        continue
                    proto = info['proto'].to(device=device, dtype=torch.float32)
                    var = info.get('var', torch.zeros_like(info['proto'])).to(device=device, dtype=torch.float32)
                    std = torch.sqrt(torch.clamp(var, min=0.0) + 1e-6) * float(getattr(args, 'replay_noise', 0.05))
                    noise = torch.randn(cls_per_class, proto.numel(), device=device) * std.view(1, -1)
                    feats.append(F.normalize(proto.view(1, -1) + noise, dim=1))
                    labels.extend([int(cls_id)] * cls_per_class)
                    class_weights.extend([float(cema_weight)] * cls_per_class)
                if len(feats) == 0:
                    return None, None, None
                return (
                    torch.cat(feats, dim=0),
                    torch.tensor(labels, dtype=torch.long, device=device),
                    torch.tensor(class_weights, dtype=torch.float32, device=device),
                )

            def _weighted_replay_ce(logits_tensor, targets_tensor, weights_tensor):
                replay_losses = F.cross_entropy(
                    logits_tensor,
                    targets_tensor,
                    reduction='none',
                )
                return (
                    replay_losses *
                    weights_tensor.to(
                        device=replay_losses.device,
                        dtype=replay_losses.dtype,
                    )
                ).mean()

            cache_device = student_classifier.fc.weight.device
            cache_dtype = student_classifier.fc.weight.dtype
            with torch.no_grad():
                expert_target_lookup = torch.zeros(
                    output_dim,
                    dtype=torch.bool,
                    device=cache_device,
                )
                expert_weight_lookup = torch.zeros(
                    output_dim,
                    dtype=cache_dtype,
                    device=cache_device,
                )
                expert_proto_lookup = torch.zeros(
                    (output_dim, num_ftrs),
                    dtype=cache_dtype,
                    device=cache_device,
                )
                for cls_id in range(output_dim):
                    expert_info = class_expert_memory.get(int(cls_id))
                    tier, cema_weight = _class_tier_decision(cls_id)
                    use_expert_target = (
                        expert_info is not None
                        and (
                            tier == 'strong'
                            or (
                                tier == 'medium'
                                and int(cls_id) in rare_present_class_set
                            )
                        )
                    )
                    if use_expert_target:
                        expert_target_lookup[cls_id] = True
                        expert_weight_lookup[cls_id] = float(cema_weight)
                        expert_proto_lookup[cls_id] = expert_info['proto'].to(
                            device=cache_device,
                            dtype=cache_dtype,
                        )

                rare_class_lookup = torch.zeros(
                    output_dim,
                    dtype=cache_dtype,
                    device=cache_device,
                )
                class_rarity_lookup = torch.ones_like(rare_class_lookup)
                class_hf_lookup = torch.zeros_like(rare_class_lookup)
                for cls_id in range(output_dim):
                    cls_count = float(label_count_map.get(int(cls_id), 0))
                    if int(cls_id) in rare_present_class_set:
                        rare_class_lookup[cls_id] = 1.0
                    class_rarity_lookup[cls_id] = 1.0 - (cls_count / max_class_count)
                    if cls_count >= local_hf_ratio * max_class_count:
                        class_hf_lookup[cls_id] = 1.0

                strong_proto_classes = sorted({
                    int(cls_id)
                    for cls_id in strong_expert_classes
                    if int(cls_id) in class_expert_memory
                })
                strong_proto_index_lookup = torch.full(
                    (output_dim,),
                    -1,
                    dtype=torch.long,
                    device=cache_device,
                )
                if len(strong_proto_classes) > 0:
                    strong_proto_matrix = torch.stack([
                        F.normalize(
                            class_expert_memory[cls_id]['proto'].to(
                                device=cache_device,
                                dtype=cache_dtype,
                            ),
                            dim=0,
                        )
                        for cls_id in strong_proto_classes
                    ], dim=0)
                    strong_proto_index_lookup[
                        torch.tensor(
                            strong_proto_classes,
                            dtype=torch.long,
                            device=cache_device,
                        )
                    ] = torch.arange(
                        len(strong_proto_classes),
                        dtype=torch.long,
                        device=cache_device,
                    )
                else:
                    strong_proto_matrix = None

            logger.info(
                'Client %d tiered class-expert training | local_present=%d | rare_present=%d',
                k,
                len(label_count_map),
                len(rare_present_class_set),
            )

            for ep in range(args.e_distill):
                loss_collector = []
                stage_ratio = float(ep + 1) / float(max(1, args.e_distill))
                eta_loc = max(0.6, 1.0 - 0.25 * stage_ratio)
                if stage_ratio <= 0.3:
                    eta_expert = 0.20
                elif stage_ratio <= 0.7:
                    eta_expert = 0.45 + 0.35 * (stage_ratio - 0.3) / 0.4
                else:
                    eta_expert = 0.80 + 0.20 * (stage_ratio - 0.7) / 0.3

                for x, target in train_dl_local:
                    x, target = x.to(device), target.to(device)
                    optimizer.zero_grad()

                    feat_s_shared = student_encoder(x)
                    feat_s = student_adapter(feat_s_shared)
                    logits_s = student_classifier(feat_s)
                    logits_s_balanced = _apply_prior_adjustment(logits_s)

                    with torch.no_grad():
                        feat_p = private_anchor_adapters[k](private_anchor_encoders[k](x))
                        logits_local_anchor = private_anchor_classifiers[k](feat_p)

                    rare_mask = rare_class_lookup.index_select(0, target).to(dtype=feat_s.dtype)
                    ce_personal_losses = F.cross_entropy(logits_s_balanced, target, reduction='none')
                    rare_ce_weight = method_round_scale * float(max(getattr(args, 'lambda_rare_ce', 1.5), 0.0))
                    loss_hard = ((1.0 + rare_ce_weight * rare_mask) * ce_personal_losses).mean()
                    feat_s_norm = F.normalize(feat_s, dim=1)
                    feat_s_shared_norm = F.normalize(feat_s_shared, dim=1)
                    feat_p_norm = F.normalize(feat_p, dim=1)

                    local_feat_losses = ((feat_s_norm - feat_p_norm) ** 2).mean(dim=1)
                    local_logit_losses = F.kl_div(
                        F.log_softmax(logits_s / local_temperature, dim=1),
                        F.softmax(logits_local_anchor / local_temperature, dim=1),
                        reduction='none',
                    ).sum(dim=1) * (local_temperature ** 2)

                    expert_proto_con_losses = _prototype_contrastive_losses(
                        feat_s_shared_norm,
                        target,
                        strong_proto_matrix,
                        strong_proto_index_lookup,
                    )

                    _need_persample = (method_round_scale > 0.0) and (
                        float(lambda_fb) > 0.0
                        or float(getattr(args, 'lambda_personal_proto', 0.3)) > 0.0
                        or float(getattr(args, 'lambda_align', 0.3)) > 0.0
                    )
                    if _need_persample:
                        global_target_present = expert_target_lookup.index_select(0, target)
                        expert_sample_weights = (
                            expert_weight_lookup.index_select(0, target).to(dtype=feat_s.dtype)
                            * global_target_present.to(dtype=feat_s.dtype)
                        )
                        global_teacher_conf = expert_sample_weights
                        target_prototypes = expert_proto_lookup.index_select(0, target).to(
                            dtype=feat_s_shared.dtype,
                        )
                        target_present_float = global_target_present.to(
                            dtype=feat_s_shared.dtype,
                        )
                        global_proto_losses = (
                            ((feat_s_shared_norm - target_prototypes) ** 2).mean(dim=1)
                            * target_present_float
                        )
                        personal_proto_losses = (
                            ((feat_s_norm - target_prototypes.to(dtype=feat_s.dtype)) ** 2).mean(dim=1)
                            * global_target_present.to(dtype=feat_s.dtype)
                        )
                    else:
                        global_proto_losses = torch.zeros_like(local_feat_losses)
                        global_target_present = torch.zeros_like(local_feat_losses, dtype=torch.bool)
                        global_teacher_conf = torch.zeros_like(local_feat_losses)
                        expert_sample_weights = torch.zeros_like(local_feat_losses)
                        personal_proto_losses = torch.zeros_like(local_feat_losses)

                    rarity = class_rarity_lookup.index_select(0, target).to(dtype=feat_s.dtype)
                    hf_mask = class_hf_lookup.index_select(0, target).to(dtype=feat_s.dtype)
                    global_conf_mask = global_target_present & (global_teacher_conf >= global_target_thresh)
                    global_gate = global_conf_mask.to(feat_s.dtype) * global_teacher_conf
                    rarity_gate = torch.clamp(0.5 + 0.5 * rarity, min=0.0, max=1.0)
                    expert_gate = torch.maximum(rare_mask, rarity_gate) * global_gate
                    lambda_loc_eff = method_round_scale * args.lambda_loc * (1.0 + 0.5 * hf_mask)
                    lambda_fb_eff = method_round_scale * lambda_fb * expert_gate

                    local_personal_losses = local_logit_losses + gamma_hf * hf_mask * local_feat_losses
                    fallback_losses = gamma_global_proto * global_proto_losses
                    personal_proto_align_losses = (
                        method_round_scale *
                        float(getattr(args, 'lambda_personal_proto', 0.3)) *
                        rare_mask *
                        expert_sample_weights *
                        personal_proto_losses
                    )

                    seen_replay_classes = [
                        int(cls_id)
                        for cls_id in rare_present_class_set
                        if _class_tier_decision(cls_id)[0] in ('strong', 'medium')
                    ]
                    replay_per_class = max(0, int(getattr(args, 'replay_per_class', 8)))
                    rare_replay_min_count = max(1, int(getattr(
                        args,
                        'rare_replay_min_count',
                        max(1, int(getattr(args, 'class_reliable_min', 8))),
                    )))
                    rare_replay_boost = max(1.0, float(getattr(args, 'rare_replay_boost', 2.0)))
                    seen_replay_counts = {}
                    for cls_id in seen_replay_classes:
                        local_cls_count = int(label_count_map.get(int(cls_id), 0))
                        cls_replay_per_class = replay_per_class
                        if 0 < local_cls_count < rare_replay_min_count:
                            cls_replay_per_class = int(ceil(float(replay_per_class) * rare_replay_boost))
                        seen_replay_counts[int(cls_id)] = cls_replay_per_class
                    seen_u, seen_y, seen_w = _sample_expert_features(seen_replay_classes, seen_replay_counts)
                    if seen_u is not None:
                        seen_h = student_adapter(seen_u.to(dtype=feat_s.dtype))
                        seen_logits = student_classifier(seen_h)
                        loss_rep_seen = _weighted_replay_ce(seen_logits, seen_y, seen_w)
                    else:
                        loss_rep_seen = torch.zeros((), dtype=feat_s.dtype, device=device)
                    loss_expert_align = (expert_sample_weights * global_proto_losses).mean()
                    loss_expert_pcon = expert_proto_con_losses.mean()

                    distill_losses = (
                        eta_loc * lambda_loc_eff * local_personal_losses +
                        eta_expert * lambda_fb_eff * fallback_losses +
                        personal_proto_align_losses
                    )

                    distill_losses = torch.nan_to_num(distill_losses, nan=0.0, posinf=10.0, neginf=0.0)
                    total_loss = (
                        loss_hard +
                        distill_losses.mean() +
                        method_round_scale * float(getattr(args, 'lambda_rep_seen', 0.7)) * loss_rep_seen +
                        method_round_scale * float(getattr(args, 'lambda_align', 0.3)) * loss_expert_align +
                        method_round_scale * float(getattr(args, 'lambda_pcon', 0.1)) * loss_expert_pcon +
                        method_round_scale * float(getattr(args, 'lambda_adapter_reg', 0.01)) * student_adapter.identity_regularizer()
                    )
                    if not torch.isfinite(total_loss):
                        logger.warning('Client %d batch skipped due to non-finite total_loss', k)
                        continue
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(student_encoder.parameters()) +
                        list(student_adapter.parameters()) +
                        list(student_classifier.parameters()),
                        max_norm=2.0,
                    )
                    grad_is_finite = True
                    for p_ in (
                        list(student_encoder.parameters()) +
                        list(student_adapter.parameters()) +
                        list(student_classifier.parameters())
                    ):
                        if p_.grad is not None and not torch.isfinite(p_.grad).all():
                            grad_is_finite = False
                            break
                    if not grad_is_finite:
                        logger.warning('Client %d batch skipped due to non-finite gradients', k)
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    optimizer.step()
                    loss_collector.append(total_loss.item())

                if len(loss_collector) > 0:
                    logger.info('Client %d Distill Epoch %d Loss: %.6f' % (k, ep, float(sum(loss_collector) / len(loss_collector))))

            student_encoder.to("cpu")
            student_adapter.to("cpu")
            student_classifier.to("cpu")
            private_anchor_encoders[k].to("cpu")
            private_anchor_adapters[k].to("cpu")
            private_anchor_classifiers[k].to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()


        logger.info("Evaluating personalized local-test accuracy on client-specific 3:1 splits...")
        avg_local_test_acc = 0.0
        total_head_correct = 0
        total_head_count = 0
        total_rare_present_correct = 0
        total_rare_present_count = 0

        for k in selected:
            encoder_list[k].to(device).eval()
            adapter_list[k].to(device).eval()
            classifier_list[k].to(device).eval()

            noise_level = args.noise if k != args.n_parties - 1 else 0
            if args.noise_type != 'space' and args.n_parties > 1:
                noise_level = args.noise / (args.n_parties - 1) * k

            test_dl_local = _get_pfedscpt_paper_loader(k, is_train=False, batch_size=32, noise_level=noise_level)

            eval_groups = _client_class_groups(k)
            head_cls = eval_groups['head']
            rare_present_cls = eval_groups['rare_present']
            correct_personal = 0
            total_eval = 0
            class_total = {cls_id: 0 for cls_id in range(output_dim)}
            class_correct = {cls_id: 0 for cls_id in range(output_dim)}

            def _predict_personal_batch(x_eval):
                feat_eval_shared = encoder_list[k](x_eval)
                feat_eval = adapter_list[k](feat_eval_shared)
                logits_personal = classifier_list[k](feat_eval)
                personal_probs = F.softmax(logits_personal, dim=1)
                personal_conf, personal_pred = torch.max(personal_probs, dim=1)
                return personal_pred, personal_conf

            with torch.no_grad():
                for x_eval, y_eval in test_dl_local:
                    x_eval = x_eval.to(device)
                    y_eval = y_eval.to(device, dtype=torch.int64)
                    pred_eval, personal_conf = _predict_personal_batch(x_eval)
                    correct_personal += int((pred_eval == y_eval).sum().item())
                    total_eval += int(y_eval.numel())
                    for cls_id in torch.unique(y_eval):
                        cls_int = int(cls_id.item())
                        cls_mask = y_eval == cls_id
                        class_total[cls_int] += int(cls_mask.sum().item())
                        class_correct[cls_int] += int((pred_eval[cls_mask] == y_eval[cls_mask]).sum().item())
            encoder_list[k].to("cpu")
            adapter_list[k].to("cpu")
            classifier_list[k].to("cpu")
            local_acc = float(correct_personal) / float(max(1, total_eval))
            avg_local_test_acc += local_acc
            head_correct = sum(int(class_correct.get(cls_id, 0)) for cls_id in head_cls)
            head_total = sum(int(class_total.get(cls_id, 0)) for cls_id in head_cls)
            rare_present_correct = sum(int(class_correct.get(cls_id, 0)) for cls_id in rare_present_cls)
            rare_present_total = sum(int(class_total.get(cls_id, 0)) for cls_id in rare_present_cls)
            client_metrics.append({"round": round, "client": int(k), "overall": local_acc,
                                   "head_correct": head_correct, "head_total": head_total,
                                   "rarep_correct": rare_present_correct, "rarep_total": rare_present_total})
            total_head_correct += head_correct
            total_head_count += head_total
            total_rare_present_correct += rare_present_correct
            total_rare_present_count += rare_present_total
            logger.info(
                "Client %d | Personal: %.4f | Head: %s | RareP: %s",
                k,
                local_acc,
                _format_count_acc(head_correct, head_total),
                _format_count_acc(rare_present_correct, rare_present_total),
            )

        avg_local_test_acc /= len(selected)
        row = {"round": round, "clients_evaluated": len(selected), "overall": avg_local_test_acc,
               "head": total_head_correct / total_head_count if total_head_count else None,
               "rarep": total_rare_present_correct / total_rare_present_count if total_rare_present_count else None,
               **{tier: sum(t == tier for t, w in class_expert_tiers.values())
                  for tier in ("strong", "medium", "weak")}}
        history.append(row)
        with (args.output_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        with (args.output_dir / "client_metrics.csv").open("a", newline="", encoding="utf-8") as stream:
            rows = client_metrics[-len(selected):]
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)
        logger.info(
            '>> pFedSCPT Round %d | Personal: %.4f | Head: %s | RareP: %s',
            round,
            avg_local_test_acc,
            _format_count_acc(total_head_correct, total_head_count),
            _format_count_acc(total_rare_present_correct, total_rare_present_count),
        )

    result = {"history": history, "memory": class_expert_memory,
              "encoders": encoder_list, "adapters": adapter_list, "classifiers": classifier_list,
              "teachers": {"encoders": private_anchor_encoders, "adapters": private_anchor_adapters,
                           "classifiers": private_anchor_classifiers}}
    if args.save_checkpoint:
        checkpoint = {"config": _json_config(args), "round": args.comm_round - 1,
                      "memory": class_expert_memory, "history": history}
        for name in ["encoders", "adapters", "classifiers"]:
            checkpoint[name] = [{k: v.detach().cpu() for k, v in model.state_dict().items()}
                                for model in result[name]]
        checkpoint["teachers"] = {
            name: {cid: {k: v.detach().cpu() for k, v in module.state_dict().items()}
                   for cid, module in modules.items()}
            for name, modules in result["teachers"].items()}
        torch.save(checkpoint, args.output_dir / "final.pt")
    return result


# ==========================================================================
# 6. Entry point and run artifacts
# ==========================================================================

def _json_config(args):
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}


def main(argv=None):
    args = get_args(argv)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run with --device cpu")
    torch.set_num_threads(args.num_threads)
    random.seed(args.init_seed)
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.init_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.output_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = Path("runs") / f"{args.dataset}_seed{args.init_seed}_{stamp}"
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in ["metrics.csv", "config.json", "training.log"]):
        raise FileExistsError("Output directory already contains a run; choose a new --output-dir")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "training.log", encoding="utf-8")],
        force=True,
    )
    config = _json_config(args)
    config["environment"] = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                             "numpy": np.__version__, "cuda": torch.version.cuda}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    split = build_pfedscpt_paper_split(args.dataset, args.datadir, args.partition, args.n_parties,
                                     beta=args.beta, seed=args.init_seed, train_ratio=0.75)
    for client in range(args.n_parties):
        if not len(split["train_position_map"][client]) or not len(split["test_position_map"][client]):
            raise ValueError(f"Client {client} has an empty split; change seed, partition, or client count")
    if len(np.unique(split["y_all"])) != NUM_CLASSES[args.dataset]:
        raise ValueError("Dataset has an unexpected number of classes")
    np.savez_compressed(args.output_dir / "partition.npz", **{
        f"{phase}_{cid}": split[f"{phase}_position_map"][cid]
        for phase in ["train", "test"] for cid in range(args.n_parties)})
    logger.info("pFedSCPT | dataset=%s | model=%s | clients=%d | device=%s",
                args.dataset, args.model, args.n_parties, args.device)
    result = run_federated(args, split)
    logger.info("Finished. Run outputs: %s", args.output_dir)
    return result


if __name__ == "__main__":
    main()
