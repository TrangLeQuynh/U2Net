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
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from tools.utils import check_size

# transform
transform=transforms.Compose([
    RescaleT(320),
    ToTensorLab(flag=0),
])

def save_output(image_name, pred, d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]

    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    imo.save(d_dir+imidx+'.png')

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

# def main():
#     args = parse_args()

#     # --------- 1. get image path and name ---------
#     # model_name=  'u2net'#u2netp
#     model_name = args.model


#     image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
#     prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
#     model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

#     img_name_list = glob.glob(image_dir + os.sep + '*')
#     print(img_name_list)

#     # --------- 2. dataloader ---------
#     #1. dataloader
#     test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
#                                         lbl_name_list = [],
#                                         transform=transforms.Compose([RescaleT(320),
#                                                                       ToTensorLab(flag=0)])
#                                         )
#     test_salobj_dataloader = DataLoader(test_salobj_dataset,
#                                         batch_size=1,
#                                         shuffle=False,
#                                         num_workers=1)

#     # --------- 3. model define ---------
#     net = build_model(args=args)

#     # --------- 4. inference for each image ---------
#     for i_test, data_test in enumerate(test_salobj_dataloader):

#         print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

#         inputs_test = data_test['image']
#         inputs_test = inputs_test.type(torch.FloatTensor)

#         if args.gpu:
#             inputs_test = Variable(inputs_test.cuda())
#         else:
#             inputs_test = Variable(inputs_test)

#         d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

#         # normalization
#         pred = d1[:,0,:,:]
#         pred = normPRED(pred)

#         # save results to test_results folder
#         if not os.path.exists(prediction_dir):
#             os.makedirs(prediction_dir, exist_ok=True)
#         save_output(img_name_list[i_test],pred,prediction_dir)

#         del d1,d2,d3,d4,d5,d6,d7    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", '-m', type=str, default="u2net") # u2netp or u2net
    parser.add_argument("--source", '-s', type=str, required=True)
    parser.add_argument("--gpu", '-g',action='store_true', default=False)
    parser.add_argument("--jit", '-j',action='store_true', default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    prediction_dir = os.path.join(os.getcwd(), 'test_data', args.model + '_results' + os.sep)

    # Define model
    net = build_model(args=args)
    
    # Inference image
    img = preprocess(args.source)["image"]
    img = img.type(torch.FloatTensor)
    if args.gpu is True:
        img = Variable(img.unsqueeze(0).cuda())
    else:
        img = Variable(img.unsqueeze(0))

    begin = time.time()
    pred= net(img)
    end = time.time()
    print("{} ms".format(1000 *(end - begin)))

    # save results to test_results folder
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    save_output(args.source, pred, prediction_dir)
