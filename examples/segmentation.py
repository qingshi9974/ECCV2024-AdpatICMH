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

from torch.utils.data import DataLoader
from torchvision import transforms

from collections import OrderedDict
import pickle


import yaml

from models.tic_sfma import TIC_SFMA

## General
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from utils.dataloader import MSCOCO, Kodak

## Test
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.data.detection_utils import read_image

from contextlib import ExitStack, contextmanager
from utils.predictor import ModPredictor
from utils.alignment import Alignment


## Function for model to eval
@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


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


class TaskLoss(nn.Module):
    def __init__(self, cfg, device) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.task_net = build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3))
        checkpoint = OrderedDict()
        with open(cfg.MODEL.WEIGHTS, 'rb') as f:
            FPN_ckpt = pickle.load(f)
            for k, v in FPN_ckpt['model'].items():
                if 'backbone' in k:
                    checkpoint['.'.join(k.split('.')[1:])] = torch.from_numpy(v)
        self.task_net.load_state_dict(checkpoint, strict=True)
        self.task_net = self.task_net.to(device)
        for k, p in self.task_net.named_parameters():
            p.requires_grad = False
        self.task_net.eval()
        self.align = Alignment(divisor=32).to(device)
        self.pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).view(-1, 1, 1).to(device)

    def forward(self, output, d, train_mode=False):
        with torch.no_grad():
            ## Ground truth for perceptual loss
            d = d.flip(1).mul(255)
            d = d - self.pixel_mean
            if not train_mode:
                d = self.align.align(d)
            gt_out = self.task_net(d)
        
        x_hat = torch.clamp(output["x_hat"], 0, 1)
        x_hat = x_hat.flip(1).mul(255)
        x_hat = x_hat - self.pixel_mean
        if not train_mode:
            x_hat = self.align.align(x_hat)
        task_net_out = self.task_net(x_hat)

        distortion_p2 = nn.MSELoss(reduction='none')(gt_out["p2"], task_net_out["p2"])
        distortion_p3 = nn.MSELoss(reduction='none')(gt_out["p3"], task_net_out["p3"])
        distortion_p4 = nn.MSELoss(reduction='none')(gt_out["p4"], task_net_out["p4"])
        distortion_p5 = nn.MSELoss(reduction='none')(gt_out["p5"], task_net_out["p5"])
        distortion_p6 = nn.MSELoss(reduction='none')(gt_out["p6"], task_net_out["p6"])

        return 0.2*(distortion_p2.mean()+distortion_p3.mean()+distortion_p4.mean()+distortion_p5.mean()+distortion_p6.mean())


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
  train_dataloader, optimizer, model, criterion_rd, criterion_task, lmbda
):
    model.train()
    device = next(model.parameters()).device
    bpps =  AverageMeter()
    total_losses = AverageMeter()
    perc_losses  = AverageMeter()
    data_times =AverageMeter()
    model_times =AverageMeter()
    time1 = time.time()
    for i, d in enumerate(train_dataloader):
        data_times.update(time.time()-time1)
        d = d.to(device)
      

        optimizer.zero_grad()
        time2= time.time()

        out_net = model(d)

        out_criterion = criterion_rd(out_net, d)
        perc_loss = criterion_task(out_net, d)
        total_loss = lmbda*perc_loss + out_criterion['bpp_loss']
        total_loss.backward()
        optimizer.step()


        total_losses.update(total_loss.item())
        perc_losses.update(perc_loss.item())

        bpps.update(out_criterion['bpp_loss'].item())
        model_times.update(time.time()-time2)
        if i%100==0:
            
            update_txt=f'[{i*len(d)}/{len(train_dataloader.dataset)}] | data time :{data_times.avg:.4f} | model time :{model_times.avg:.4f} | Loss: {total_losses.avg:.3f} |  perc loss: {perc_losses.avg:.5f} | Bpp loss: {bpps.avg:.3f} '
            print(datetime.now(),update_txt)
        time1 = time.time()
    print(update_txt)

def validation_epoch(epoch, val_dataloader, model, criterion_rd, criterion_task, lmbda):
    model.eval()
    device = next(model.parameters()).device

    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    percloss = AverageMeter()
    totalloss = AverageMeter()

    with torch.no_grad():
        tqdm_meter = enumerate(val_dataloader)
        for i, d in tqdm_meter:
        
            align = Alignment(divisor=256, mode='pad',padding_mode='constant').to(device)
            #divisor: 256 for TIC,  64 for mbt2018mean and cheng2020anchor

            d = d.to(device)
            align_d = align.align(d)

            out_net = model(align_d)
            # print(d.shape)
            out_net['x_hat'] = align.resume(out_net['x_hat']).clamp_(0, 1)
            out_criterion = criterion_rd(out_net, d)
            perc_loss = criterion_task(out_net, d)
            total_loss = lmbda*perc_loss + out_criterion['bpp_loss']

            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion['psnr'])
            percloss.update(perc_loss)
            totalloss.update(total_loss)

        txt = f"Loss: {totalloss.avg:.3f} | MSE loss: {mse_loss.avg:.5f} | Perception loss: {percloss.avg:.4f} | Bpp loss: {bpp_loss.avg:.4f}"
    print(txt)
    model.train()
    print(f"Epoch: {epoch} | bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f}")
    return totalloss.avg


def test_epoch(test_dataloader, model, criterion_rd, criterion_task,  predictor, evaluator):
    model.eval()
    device = next(model.parameters()).device
    pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).view(-1, 1, 1).to(device)

    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    feat_loss =AverageMeter()

    with torch.no_grad():
        tqdm_meter = enumerate(test_dataloader)
        for i, batch in tqdm_meter:
            with ExitStack() as stack:
                ## model to eval()
                if isinstance(predictor.model, nn.Module):
                    stack.enter_context(inference_context(predictor.model))
                stack.enter_context(torch.no_grad())    

                align = Alignment(divisor=256, mode='pad',padding_mode='constant').to(device)
                #divisor: 256 for TIC,  64 for mbt2018mean and cheng2020anchor
                rcnn_align = Alignment(divisor=32).to(device)
                img = read_image(batch[0]["file_name"], format="BGR")
                d = torch.stack([batch[0]['image'].float().div(255)]).flip(1).to(device)
                align_d = align.align(d)
   
                out_net = model(align_d)
                out_net['x_hat'] = align.resume(out_net['x_hat']).clamp_(0, 1)
                out_criterion = criterion_rd(out_net, d)
                perc_loss = criterion_task(out_net, d)
          
                feat_loss.update(perc_loss.item())
                trand_y_tilde = out_net['x_hat'].flip(1).mul(255)
                trand_y_tilde = rcnn_align.align(trand_y_tilde - pixel_mean)

                bpp_loss.update(out_criterion["bpp_loss"])
            
                psnr.update(out_criterion['psnr'])

                ## MaskRCNN
                predictions = predictor(img, trand_y_tilde)
                # for i in range(5000):
                evaluator.process(batch, [predictions])
            txt = f"|{i*len(d)}/{len(test_dataloader.dataset)}] |Bpp loss: {bpp_loss.avg:.4f} | PSNR loss: {psnr.avg:.4f}"
     
            if i%500==0:
                print(datetime.now(),txt)
    print(f"bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f}")

    results = evaluator.evaluate()
    model.train()
    return


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    logging.info(f"Saving checkpoint: {base_dir+filename}")
    torch.save(state, base_dir+filename)
    if is_best:
        logging.info(f"Saving BEST checkpoint: {base_dir+filename}")
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

    parser.add_argument("-T", "--TEST", action='store_true', help='Testing')
    parser.add_argument(
        "--gpu_id",
        type=int,
        default="0",
        help="Path to config file",
    )
    parser.add_argument(
        "--quality_level",
        type=int,
        default="1",
        help="Path to config file",
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

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.dataset=='coco':
        cfg = get_cfg() # get default cfg
        cfg.merge_from_file("./config/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = args.maskrcnn_path
    
        det_transformer = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        ## Training
        train_dataset = MSCOCO(args.dataset_path+"/train2017/",
                               det_transformer,
                               "./examples/utils/img_list.txt")
        val_dataset = Kodak(args.dataset_path+"/Kodak/", transforms.ToTensor())

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True,
                                      pin_memory=(device=="cuda"))
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=args.test_batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    pin_memory=(device=="cuda"))
        
        ## Testing
        # if args.TEST:
        json_path = args.dataset_path + "/annotations/instances_val2017.json"
        image_path = args.dataset_path + "/val2017"
        register_coco_instances("compressed_coco", {}, json_path, image_path)
        evaluator = COCOEvaluator("compressed_coco", cfg, False, output_dir="./coco_log")
     
        evaluator.reset()
        test_dataloader = build_detection_test_loader(cfg, "compressed_coco")
    
        cfg.MODEL.META_ARCHITECTURE = 'GeneralizedRCNN_with_Rate'
        predictor = ModPredictor(cfg)
    net = TIC_SFMA(N=128,M=192)

    net = net.to(device)
    print('total paramaters:',sum(p.numel() for p in net.parameters() )/1e6)

    for k, p in net.named_parameters():
        if "sfma" not in k:
            p.requires_grad = False    
    print('tuning paramaters:',sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6)

    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
    rdcriterion = RateDistortionLoss()
    taskcriterion = TaskLoss(cfg, device)

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
        test_epoch(test_dataloader, net, rdcriterion, taskcriterion,predictor, evaluator)
        return

    best_loss = validation_epoch(-1, val_dataloader, net, rdcriterion, taskcriterion, args.VPT_lmbda)
    tqrange = tqdm.trange(last_epoch, args.epochs)
    for epoch in tqrange:
        print('/')
        print('Epoch:',epoch)
        train_one_epoch(train_dataloader, optimizer, net, rdcriterion, taskcriterion, args.VPT_lmbda)
        loss = validation_epoch(epoch, val_dataloader, net, rdcriterion, taskcriterion, args.VPT_lmbda)
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

 
        evaluator.reset()
        test_epoch(test_dataloader, net, rdcriterion, taskcriterion,predictor, evaluator)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    main(sys.argv[1:])
