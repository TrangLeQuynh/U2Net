import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
import argparse
import time
import os
import os.path as osp
import cv2
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from tools.utils import check_size
from tqdm import tqdm
from tools.saleval import SalEval

# transform
transform=transforms.Compose([
    RescaleT(320),
    ToTensorLab(flag=0),
])

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def preprocess(img_path):
    image = io.imread(img_path)
    imidx = np.array([0])

    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if(3==len(label_3.shape)):
        label = label_3[:,:,0]
    elif(2==len(label_3.shape)):
        label = label_3

    if(3==len(image.shape) and 2==len(label.shape)):
        label = label[:,:,np.newaxis]
    elif(2==len(image.shape) and 2==len(label.shape)):
        image = image[:,:,np.newaxis]
        label = label[:,:,np.newaxis]

    sample = {'imidx':imidx, 'image':image, 'label':label}
    return transform(sample)

def build_model(args):
    # model_name=  'u2net'#u2netp
    model_name = args.model
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    if args.gpu is True:
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    if args.jit is True:
        net_jit_path = os.path.join('saved_models', model_name, model_name + '_jit.pt')
        if os.path.isfile(net_jit_path) is False:
            input = torch.rand(1, 3, 320, 320)
            net = torch.jit.trace(net, input.cuda() if args.gpu is True else input, strict=False)
            torch.jit.save(net, net_jit_path)
        else:
            del net
            net = torch.jit.load(net_jit_path)
    check_size(model=net)
    return net

@torch.no_grad()
def test(args, model, image_list, label_list, save_dir):
    eval = SalEval()
    for idx in tqdm(range(len(image_list))):
        label = cv2.imread(label_list[idx], 0)
        label = label / 255
        label = torch.from_numpy(label).float().unsqueeze(0)

        if args.gpu is True:
            label = label.cuda()
        # Inference image
        img = preprocess(image_list[idx])["image"]
        img = img.type(torch.FloatTensor)
        if args.gpu is True:
            img = Variable(img.unsqueeze(0).cuda())
        else:
            img = Variable(img.unsqueeze(0))

        d1,d2,d3,d4,d5,d6,d7= model(img)
        d1 = F.interpolate(d1, size=label.shape[1:], mode='bilinear', align_corners=False)
        
        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        del d1,d2,d3,d4,d5,d6,d7

        eval.add_batch(pred.cuda(), label.unsqueeze(dim=0).cuda())

        predict_np = pred.squeeze().cpu().data.numpy()
        img_out = Image.fromarray(predict_np*255).convert('RGB')
        img_out.save(osp.join(save_dir, osp.basename(image_list[idx])[:-4] + '.png'))

    F_beta, MAE = eval.get_metric()
    print('Overall F_beta (Val): %.4f\t MAE (Val): %.4f' % (F_beta, MAE))

def main(args, file_list):
    data_dir = args.root
    savedir = f"outputs"
    # read all the images in the folder
    image_list = list()
    label_list = list()
    with open(data_dir + '/' + file_list + '.txt') as fid:
        for line in fid:
            line_arr = line.split()
            image_list.append(data_dir + '/' + line_arr[0].strip())
            label_list.append(data_dir + '/' + line_arr[1].strip())

    # Define model
    model = build_model(args=args)

    save_dir = savedir + '/' + file_list + '/'
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    test(args, model, image_list, label_list, save_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", '-m', type=str, default="u2netp") # u2netp or u2net
    parser.add_argument("--root", type=str, default="/home/timebird3/Documents/AI/model_template/Salient-Object-Detection/EDN/data")
    parser.add_argument("--gpu", '-g',action='store_true', default=False)
    parser.add_argument("--jit", '-j',action='store_true', default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # main(args, 'PASCAL-S')
    data_lists = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'PASCAL-S', 'HKU-IS', 'SOD']
    for data_list in data_lists:
        print(f"____________________{data_list}_____________________")
        main(args, data_list)
