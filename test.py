import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
from models import *
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt


def load_args():
    parser = argparse.ArgumentParser(description='ES3Net')
    parser.add_argument('--KITTI', default='2015',
                        help='KITTI version')
    parser.add_argument('--data_path', default=None,
                        help='path to test data')
    parser.add_argument('--load_cpt_path', default=None,
                        help='path for loading model checkpoint')    
    parser.add_argument('--imgL_path', default=None,
                        help='left image pair')
    parser.add_argument('--imgR_path', default= None,
                        help='right image pair')
    parser.add_argument('--save_path', default='./results',
                        help='path for saving disparity map')                               
    parser.add_argument('--model', default='RealTimeStereo',
                        choices=['RealTimeStereo', 
                                'stackhourglass', 
                                'stackhourglass2d', 
                                'basic'],
                        help='select model (default: RealTimeStereo)')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('multi_scale', action='store_true', default=False,
                        help='model used multi-scale reconstruction for training (default: False)')
    args = parser.parse_args()
    return args

def read_image_pair(imgL_path, imgR_path):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    

    imgL = Image.open(imgL_path).convert('RGB')
    imgR = Image.open(imgR_path).convert('RGB')

    imgL = infer_transform(imgL)
    imgR = infer_transform(imgR) 
    
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    

    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

    return imgL, imgR, top_pad, right_pad


def test_single_pair(model,imgL,imgR, use_cuda):
    model.eval()

    if use_cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()     

    with torch.no_grad():
        disp = model(imgL,imgR)[0]

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp

def save_pred_disp(pred_disp, save_path, basename):
    os.makedirs(save_path, exist_ok=True)
    pred_disp = (pred_disp*256).astype('uint16')
    pred_disp = Image.fromarray(pred_disp)
    pred_disp.save(os.path.join(save_path, basename+'_test.png'))
    plt.imsave(os.path.join(save_path, basename+'_plot.png'), pred_disp, cmap='plasma')

def save_pred_disp_split_dir(pred_disp, save_path, basename):
    os.makedirs(os.path.join(save_path, 'test'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'plot'), exist_ok=True)
    pred_disp = (pred_disp*256).astype('uint16')
    pred_disp = Image.fromarray(pred_disp)
    pred_disp.save(os.path.join(save_path,'test', basename+'.png'))
    plt.imsave(os.path.join(save_path, 'plot', basename+'.png'), pred_disp, cmap='plasma')

if __name__ == '__main__':
    args = load_args()
    os.makedirs(args.save_path, exist_ok=True)

    SINGLE_PAIR = False
    if args.imgL_path is not None and args.imgR_path is not None:
        SINGLE_PAIR = True
    
    MULTI_PAIRS = False
    if args.data_path is not None and\
          os.path.join(args.data_path, 'RGB_left') is not None and\
              os.path.join(args.data_path, 'RGB_right') is not None:
        MULTI_PAIRS = True

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('init model...')
    if args.model == 'RealTimeStereo':
        from models import RTStereoNet as stereoNet
    elif args.model == 'PSMNet':
        from models import PSMNet as stereoNet
    elif args.model == 'PSMNet2d':
        from models import PSMNet2d as stereoNet
    elif args.model == 'basic':
        from models import basic as stereoNet
    else:
        print('no model')

    model = stereoNet(args.maxdisp, args.multi_scale)

    if args.cuda:
        model.cuda()

    if args.load_cpt_path is not None:
        print(f'loading model from {args.load_cpt_path} ...')
        state_dict = torch.load(args.load_cpt_path)
        from collections import OrderedDict
        model_state_dict = OrderedDict()

        for k, v in state_dict['state_dict'].items():
            k = k.replace('module.', '')
            model_state_dict[k] = v

        
        model.load_state_dict(model_state_dict)

    model.eval()

    if SINGLE_PAIR:
        imgL, imgR, top_pad, right_pad = read_image_pair(args.imgL_path, args.imgR_path)

        basename = os.path.basename(args.imgL_path)
        basename = basename.replace('.png', '')
        start_time = time.time()
        pred_disp = test_single_pair(model, imgL,imgR, args.cuda)
        print('time = %.3f' %(time.time() - start_time))

        if top_pad !=0 and right_pad != 0:
            pred_disp = pred_disp[top_pad:,:-right_pad]
        elif top_pad ==0 and right_pad != 0:
            pred_disp = pred_disp[:,:-right_pad]
        elif top_pad !=0 and right_pad == 0:
            pred_disp = pred_disp[top_pad:,:]
        else:
            pred_disp = pred_disp

        pred_disp[pred_disp<0]=1e-8

        save_pred_disp(pred_disp, args.save_path, basename)

        print(f'estimated disparity map is saved at {args.save_path}')

    if MULTI_PAIRS:
        from tqdm import tqdm
        left_img_list = glob(os.path.join(args.data_path, 'RGB_left', '*.png'))
        pbar = tqdm(left_img_list)
        for imgL_path in pbar:
            imgR_path = imgL_path.replace('RGB_left', 'RGB_right')
            imgL, imgR, top_pad, right_pad = read_image_pair(imgL_path, imgR_path)

            basename = os.path.basename(imgL_path)
            basename = basename.replace('.png', '')

            start_time = time.time()
            pred_disp = test_single_pair(model, imgL,imgR, args.cuda)
            pbar.set_description(f'{basename} time = %.3f' %(time.time() - start_time))

            if top_pad !=0 and right_pad != 0:
                pred_disp = pred_disp[top_pad:,:-right_pad]
            elif top_pad ==0 and right_pad != 0:
                pred_disp = pred_disp[:,:-right_pad]
            elif top_pad !=0 and right_pad == 0:
                pred_disp = pred_disp[top_pad:,:]
            else:
                pred_disp = pred_disp

            save_pred_disp_split_dir(pred_disp, args.save_path, basename)

        print(f'all estimated disparity maps are saved at {args.data_path}!')