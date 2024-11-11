import random
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.zoo import image_models
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import yaml
from datetime import datetime
import argparse
import sys
import random
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from utils.alignment import Alignment
import math


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

def init(args):
    base_dir = f'{args.root}/{args.exp_name}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)
    if args.bitmapNum:
        os.makedirs( os.path.join(base_dir, "bit_map"), exist_ok=True)
    return base_dir


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

    parser.add_argument(
        "--bitmapNum",
        type=int,
        default=50
    )

    args = parser.parse_args(remaining)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    cls_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )

    if args.dataset=='imagenet':
        test_dataset = torchvision.datasets.ImageNet(args.dataset_path, split='val', transform=cls_transforms)
        test_dataloader = DataLoader(test_dataset,batch_size=1,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
    elif args.dataset=='coco':
        cfg = get_cfg() # get default cfg
        cfg.merge_from_file("./config/faster_rcnn_R_50_FPN_3x.yaml")
        json_path = args.dataset_path + "/annotations/instances_val2017.json"
        image_path = args.dataset_path + "/val2017"
        register_coco_instances("compressed_coco", {}, json_path, image_path)
        test_dataloader = build_detection_test_loader(cfg, "compressed_coco")
    else:
        raise NotImplementedError()

    
    if args.model == "tic_hp":
        model = image_models[args.model](quality=int(args.quality_level))
    else:
        model = image_models[args.model](quality=int(args.quality_level), prompt_config=args)
    model = model.to(device)


    if args.checkpoint: 
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if list(checkpoint["state_dict"].keys())[0][:7]=='module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:] 
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint['state_dict']
        model.load_state_dict(new_state_dict, strict=True if args.TEST else False)
    else:
        raise FileNotFoundError("No checkpoint.")
    

    rdcriterion = RateDistortionLoss(lmbda=args.lmbda)
    model.eval()
    device = next(model.parameters()).device

    if args.dataset=='imagenet': 
        with torch.no_grad():
            tqdm_meter = tqdm.tqdm(enumerate(test_dataloader),leave=False, total=len(test_dataloader))
            for i, d in tqdm_meter:

                d = d[0].to(device)
                out_net = model(d)
                out_criterion = rdcriterion(out_net, d)

                if i < args.bitmapNum:
                    bitmap = (torch.log(out_net['likelihoods']['y'][0]) / (-math.log(2))).mean(dim=0).cpu().numpy()
                    bitmap = sns.heatmap(bitmap, cmap="viridis", vmin=0, vmax=1.55)
                    plt.axis('off')
                    plt.savefig(os.path.join(base_dir, f"{str(i).zfill(6)}_BitMap.jpg"), dpi=300, bbox_inches="tight")
                    plt.clf()

                    recon = out_net['x_hat'][0].squeeze().permute(1,2,0).clamp(min=0, max=1).cpu().numpy()
                    plt.axis('off')
                    plt.imsave(os.path.join(base_dir, f"{str(i).zfill(6)}_Recon.jpg"), recon, dpi=300)
                    plt.clf()
                else:
                    break
    elif args.dataset=='coco':
        with torch.no_grad():
            tqdm_meter = tqdm.tqdm(enumerate(test_dataloader),leave=False, total=len(test_dataloader))
            for i, batch in tqdm_meter:
                align = Alignment(divisor=256, mode='resize').to(device)
                d = torch.stack([batch[0]['image'].float().div(255)]).flip(1).to(device)
                align_d = align.align(d)

                out_net = model(align_d)
                out_net['x_hat'] = align.resume(out_net['x_hat']).clamp_(0, 1)
                out_criterion = rdcriterion(out_net, d)

                if i < args.bitmapNum:
                    bitmap = (torch.log(out_net['likelihoods']['y'][0]) / (-math.log(2))).mean(dim=0).cpu().numpy()
                    bitmap = sns.heatmap(bitmap, cmap="viridis", vmin=0, vmax=1.55)
                    plt.axis('off')
                    plt.savefig(os.path.join(base_dir, f"{str(i).zfill(6)}_BitMap.jpg"), dpi=300, bbox_inches="tight")
                    plt.clf()

                    recon = out_net['x_hat'][0].squeeze().permute(1,2,0).clamp(min=0, max=1).cpu().numpy()
                    plt.axis('off')
                    plt.imsave(os.path.join(base_dir, f"{str(i).zfill(6)}_Recon.jpg"), recon, dpi=300)
                    plt.clf()
                else:
                    break
                
    else:
        raise NotImplementedError()
                

if __name__=="__main__":
    main(sys.argv[1:])
