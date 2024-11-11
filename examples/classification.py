# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from cmath import exp
import tqdm
import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from compressai.zoo import mbt2018_mean
from models.tic_sfma import TIC_SFMA

import yaml

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["rdloss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)
        return out


class FeatureHook():
    def __init__(self, module):
        module.register_forward_hook(self.attach)
    
    def attach(self, model, input, output):
        self.feature = output

class Clsloss(nn.Module):
    def __init__(self, device, perceptual_loss=False) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.classifier = resnet50(True)
        self.classifier.requires_grad_(False)
        self.hooks = [FeatureHook(i) for i in [
            self.classifier.layer1,
            self.classifier.layer2,
            self.classifier.layer3,
            self.classifier.layer4,
        ]]
        self.classifier = self.classifier.to(device)
        for k, p in self.classifier.named_parameters():
            p.requires_grad = False
        self.classifier.eval()
        self.perceptual_loss = perceptual_loss
        self.transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def forward(self, output, d, y_true):
        x_hat = torch.clamp(output["x_hat"],0,1)
        pred = self.classifier(self.transform(x_hat))
        loss = self.ce(pred, y_true)
        accu = sum(torch.argmax(pred,-1)==y_true)/pred.shape[0]
        if self.perceptual_loss:
            pred_feat = [i.feature.clone() for i in self.hooks]
            _ = self.classifier(self.transform(d))
            ori_feat = [i.feature.clone() for i in self.hooks]
            perc_loss = torch.stack([nn.functional.mse_loss(p,o, reduction='none').mean((1,2,3)) for p,o in zip(pred_feat, ori_feat)])
            perc_loss = perc_loss.mean()
            return loss, accu, perc_loss

        return loss, accu, None

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'{args.root}/{args.exp_name}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    """Set optimizer for only the parameters for propmts"""


    parameters = {
        k
        for k, p in net.named_parameters()
        if "sfma" in k
    }

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer

def train_one_epoch(
    model, criterion_rd, criterion_cls, train_dataloader, optimizer, lmbda
):
    model.train()
    device = next(model.parameters()).device
    bpps =  AverageMeter()
    accus =  AverageMeter()
    total_losses = AverageMeter()
    perc_losses  = AverageMeter()
    data_times =AverageMeter()
    model_times =AverageMeter()
    start_time = time.time()
    for i, (d,l) in enumerate(train_dataloader):
        data_times.update(time.time()-start_time)
        d = d.to(device)
        l = l.to(device)
        model_time= time.time()
        optimizer.zero_grad()
        
        out_net = model(d)

        out_criterion = criterion_rd(out_net, d)
        loss, accu, perc_loss = criterion_cls(out_net, d, l)
      
        total_loss = lmbda*perc_loss + out_criterion['bpp_loss']
        total_loss.backward()
        optimizer.step()
 
       
        model_times.update(time.time()-model_time)
        if i%20==0:
            accus.update(accu.item())
            total_losses.update(total_loss.item())
            perc_losses.update(perc_loss.item())
            bpps.update(out_criterion['bpp_loss'].item())
        if i%100==0 or i ==len(train_dataloader)-1:
            update_txt=f'[{i*len(d)}/{len(train_dataloader.dataset)}] | data time :{data_times.avg:.4f} | model time :{model_times.avg:.4f} |Loss: {total_losses.avg:.3f} | perc loss: {perc_losses.avg:.5f} | Bpp loss: {bpps.avg:.4f} | Accs: {accus.avg:.3f}'
            print(datetime.now(),update_txt)
        start_time = time.time()

    
def test_epoch(epoch, test_dataloader, model, criterion_rd, criterion_cls, lmbda, stage='test'):
    model.eval()
    device = next(model.parameters()).device

    loss_am = AverageMeter()
    percloss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    accuracy = AverageMeter()
    totalloss = AverageMeter()

    with torch.no_grad():
        for i, (d,l) in enumerate(test_dataloader):
            d = d.to(device)
            l = l.to(device)
            out_net = model(d)
            out_criterion = criterion_rd(out_net, d)
            loss, accu, perc_loss = criterion_cls(out_net, d, l)
            total_loss = lmbda*perc_loss + out_criterion['bpp_loss']

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss_am.update(loss)
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion['psnr'])
            accuracy.update(accu)
            percloss.update(perc_loss)
            totalloss.update(total_loss)

            if i%100==0 or i ==len(test_dataloader)-1:
                print(datetime.now(),f'[{i*len(d)}/{len(test_dataloader.dataset)}]',f"{epoch}  total loss:{totalloss.avg:.3f}| bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f} | accu: {accuracy.avg:.5f}")
 
    model.train()
    print(f"{epoch} total loss:{totalloss.avg:.3f}| bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f} | accu: {accuracy.avg:.5f}")
    return  totalloss.avg

def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/vpt_default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    given_configs, remaining = parser.parse_known_args(argv)
    with open(given_configs.config) as file:
        yaml_data= yaml.safe_load(file)
        parser.set_defaults(**yaml_data)
    
    parser.add_argument(
        "-T",
        "--TEST",
        action='store_true',
        help='Testing'
    )

    args = parser.parse_args(remaining)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    cls_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),		
    transforms.ToTensor()				
])

    if args.dataset=='imagenet':
        train_dataset = torchvision.datasets.ImageNet(args.dataset_path,split='train', transform=train_transforms)
        test_dataset = torchvision.datasets.ImageNet(args.dataset_path,split='val', transform=cls_transforms)
        val_dataset = test_dataset
        small_train_datasets = torch.utils.data.random_split(train_dataset,[80000]*16+[1167])

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    val_dataloader = DataLoader(val_dataset,batch_size=args.test_batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
    test_dataloader = DataLoader(test_dataset,batch_size=args.test_batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)

    net = TIC_SFMA(N=128,M=192)
    net = net.to(device)
    # net.load_state_dict(mbt2018_mean(quality=4 , pretrained=True).state_dict(),strict=False)
    print('total paramaters:',sum(p.numel() for p in net.parameters() )/1e6)
 

    for k, p in net.named_parameters():
        if "sfma" not in k :
            p.requires_grad = False
    print('tuning paramaters:',sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6)

    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.5)
    rdcriterion = RateDistortionLoss(lmbda=args.task_lmbda)
    clscriterion = Clsloss(device, True)

    last_epoch = 0
    if args.checkpoint: 
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if list(checkpoint["state_dict"].keys())[0][:7]=='module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:] 
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint['state_dict']
        net.load_state_dict(new_state_dict, strict=True if args.TEST else False)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    
    if args.TEST:
        best_loss = float("inf")
        tqrange = tqdm.trange(last_epoch, args.epochs)
        loss = test_epoch(-1, test_dataloader, net, rdcriterion,clscriterion, args.task_lmbda,'test')
        return

    best_loss = float("inf")
    tqrange = tqdm.trange(last_epoch, args.epochs)
    for epoch in tqrange:
        print(' ')
        train_dataloader = DataLoader(
        small_train_datasets[epoch%16],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
        )
        train_one_epoch(
            net,
            rdcriterion,
            clscriterion,
            train_dataloader,
            optimizer,
            args.task_lmbda
        )
        loss = test_epoch(epoch, val_dataloader, net, rdcriterion,clscriterion, args.task_lmbda,'val')
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        print('best_loss:',best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                base_dir,
                filename='checkpoint.pth.tar'
            )


if __name__ == "__main__":
    main(sys.argv[1:])
