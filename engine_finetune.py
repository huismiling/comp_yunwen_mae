# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
from random import randint, random
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

def build_targets(embeddings):
    bsize, fsize = embeddings.size()
    labels = embeddings[:5, :]
    samples = embeddings

    labels = labels.repeat(bsize,1,1)
    samples = samples.unsqueeze(1).repeat(1,5,1)
    sims = F.cosine_similarity(labels, samples, dim=2)
    # scores = F.softmax(sims, 1)

    targets = sims.argmax(1)
    targets = torch.cat([torch.range(0,4, dtype=targets.dtype,device=labels.device), targets[5:]])
    mask = sims.max(dim=1)[0] > 0.5
    mask = torch.cat([torch.ones((5,), dtype=mask.dtype, device=mask.device), mask[5:]])
    return targets, mask

def cross_output(embeddings):
    labels = embeddings[:5, :]
    samples = embeddings[:5, :]
    labels = labels.repeat(5,1,1)
    samples = samples.unsqueeze(1).repeat(1,5,1)
    sims = F.cosine_similarity(labels, samples, dim=2)
    sims = F.softmax(sims.view(sims.size(0), 5), 1)
    return sims

class god_transform(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        xp = randint(1, 6)
        yp = randint(1, 6)
        # print(x.shape)
        return x.repeat(1, yp, xp)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, god_imgs=None, log_writer=None,
                    args=None):
    # transform_train = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     god_transform(),
    #     transforms.Resize((args.input_size, args.input_size)),  # 3 is bicubic
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # god_imgs = god_imgs.to(device, non_blocking=True)

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # trans_god = []
        # for itg in god_imgs:
        #     trans_god.append(transform_train(itg).unsqueeze(0))
        # trans_god = torch.cat(trans_god).to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        # samples = torch.cat([trans_god, samples])
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs, out_embedding = model(samples)
            # with torch.no_grad():
            #     targets, mask = build_targets(out_embedding)
            # outputs = torch.cat([cross_output(out_embedding), outputs[5:, ]])
            #     # real_datas = torch.sum(mask).item()
            #     # print(real_datas)
            # if (epoch == 0 and data_iter_step < 800) or data_iter_step==0:
            #     outputs = outputs[:5]
            #     targets = targets[:5]
            #     mask    = mask[:5]
            #     print(outputs, targets)
            # loss = criterion(outputs[mask], targets[mask])
            # print(outputs.size(), targets.size())
            loss = criterion(outputs, targets)
        # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('finetune/loss', loss_value_reduce, epoch_1000x)
            # log_writer.add_scalar('finetune/acc1', acc1, epoch_1000x)
            log_writer.add_scalar('finetune/lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, god_imgs=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        god_transform(),
        transforms.Resize((224, 224)),  # 3 is bicubic
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    god_imgs = god_imgs.to(device, non_blocking=True)

    # fres = open("test_result.txt", "w+")
    results = []
    values = []
    embeddings = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        # target = batch[-1]
        images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        images = torch.cat([god_imgs, images])

        # compute output
        with torch.cuda.amp.autocast():
            output, out_embedding = model(images)
            # with torch.no_grad():
            #     target, mask = build_targets(out_embedding)
            # loss = criterion(output, target)
            output = torch.nn.functional.softmax(output, dim=1)
        value, output = output.max(dim = 1)
        # print(output.size())
        results += output.cpu().numpy()[5:].reshape(-1).tolist()
        values += value.cpu().numpy()[5:].reshape(-1).tolist()
        embeddings += out_embedding.cpu().numpy()[5:].reshape(-1).tolist()

        # fres.writelines("\n".join(map(str, results)))
        # fres.write("\n")
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # batch_size = images.shape[0]
        # metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return results, values, embeddings


if __name__ == "__main__":
    embeddings = torch.randn(100, 8)
    targets = build_targets(embeddings)

