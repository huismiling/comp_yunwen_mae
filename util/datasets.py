# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import random
from random import randint
import PIL
from PIL import Image

import torch
import torch.utils.data as data
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


trans_one = transforms.Compose([
    transforms.RandomRotation(180),
    #transforms.RandomAffine(20),
    #transforms.RandomInvert(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAdjustSharpness(2),
    transforms.RandomResizedCrop(224, scale=(0.5, 1.2)),  # 3 is bicubic
    # transforms.RandomErasing(),
    transforms.ToTensor(),
    transforms.RandomErasing(),
    ])

class CustomDataset(data.Dataset):
    """
    Custom Dataset for yunwen competiton.
    It is necessary to override the ``__getitem__`` and ``__len__`` method.

    Args:
        root (string): Root directory of dataset.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    .. note::

        :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
    """

    _repr_indent = 4

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        finetune:Optional[bool] = True,
        godden_root = "./",
    ) -> None:
        # super().__init__(root, transform=transform, target_transform=target_transform)
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = "/"
        self.godden_root = godden_root
        self.flist = []
        self.flabel = []
        if os.path.isdir(root):
            self.root = root
            self.flist = sorted(os.listdir(self.root))
            self.flabel = [-1] * len(self.flist)
        elif os.path.isfile(root):
            for itf in open(root).readlines(): 
                itsp = itf.strip().split(",")
                if len(itsp) == 2:
                    fpath, label = itsp[0], itsp[1]
                else:
                    fpath, label = itsp[0], -1
                self.flist.append(fpath)
                self.flabel.append(int(label))
        
        self.len_flist = len(self.flist)
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        self.transforms = transforms
        self.finetune = finetune

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        # path = os.path.join(self.root, self.flist[index])
        label = index % 5
        use_gods = (self.finetune and (randint(0, 1) == 0)) or self.len_flist==0
        if use_gods:
            path = os.path.join(self.godden_root, "godden/yunwen{}.png".format(label))
        else:
            fidx = index % self.len_flist
            path = os.path.join(self.root, self.flist[fidx])
            label = self.flabel[fidx]
            # path = '/diskssd0/datasets/yunwen/train/65317.png'
        with open(path, "rb") as f:
            img = Image.open(f)
            sample = img.convert("RGB")
        # print(label, path)
        xp = randint(1, 6)
        yp = randint(1, 6)
        trans_samples = []
        if use_gods:
            if randint(0, 4) == 0:
                for itx in range(xp):
                    ys = [trans_one(sample)] if itx==0 else [torch.zeros_like(trans_one(sample))]
                    for ity in range(1, yp):
                        img = trans_one(sample)
                        if randint(0, 4) == 0:
                            ys.append(img)
                        else:
                            ys.append(torch.zeros_like(img))
                    random.shuffle(ys)
                    trans_samples.append(torch.cat(ys, dim=2))
                random.shuffle(trans_samples)
                sample = torch.cat(trans_samples, dim=1)
            else:
                sample = trans_one(sample)
        elif self.finetune:
            sample = trans_one(sample)

        if self.transform is not None:
            
            sample = self.transform(sample)
        
        return sample, label

    def __len__(self) -> int:
        if self.finetune:
            return 20000 + len(self.flist)
        else:
            return len(self.flist)

    def __repr__(self) -> str:
        head = "Dataset yunwen"
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        return ""
