import os
import argparse
from tqdm import tqdm
from typing import List
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


from model.simplenet import SimpleNet
from dataset import FullDataset, CropDataset
from metrics import psnr, ssim

def save_compare_img(img_list:List[np.ndarray], filename, save_dir):
    """function to save image

    Args:
        img_list (List[np.ndarray]): [target, input, restored] images's list, target, input, resotred are np.ndarray and have (N, C, H, W) shape
    """
    concat_img = np.concatenate(img_list, axis=3) # (N, C, H, 3W)
    # transpose to (N, H, W, C) to save with plt
    concat_img = np.transpose(concat_img, (0, 2, 3, 1))
    for i, batch in enumerate(concat_img): # batch (N, H, W, C)
        plt.imsave(os.path.join(save_dir, 'imgs', filename[i]), batch)

def save_restored_img(restored_img:np.ndarray, filename, save_dir):
    img = np.transpose(restored_img, (0, 2, 3, 1))
    for i, batch in enumerate(img):
        plt.imsave(os.path.join(save_dir, 'restored_imgs', filename[i]), batch)
    
def load_checkpoint(model, weights_path):
    chkpoint = torch.load(weights_path)
    print('It''s %d epoch weights' % (chkpoint['epoch']))
    print("### Loading weights ###")
    model.load_state_dict(chkpoint['network'])
    
def test(model, opt):
    test_data = FullDataset(opt.data_dir)
    testloader = DataLoader(test_data, 1, shuffle=False)
    device = torch.device('cuda:'+opt.gpu) if opt.gpu != '-1' else torch.device('cpu')
    save_dir = os.path.join(opt.save_dir, f'{model.__class__.__name__}' + str(len(os.listdir(opt.save_dir))))
    os.makedirs(save_dir)
    
    load_checkpoint(model, opt.weights)
    model = model.to(device)
    if opt.save_txt:
        f = open(os.path.join(save_dir, 'results.txt'), 'w')
        f.write(f"{opt.data_dir}, ")
    if opt.save_img:
        os.mkdir(os.path.join(save_dir, 'imgs'))
        os.mkdir(os.path.join(save_dir, 'restored_imgs'))
    
    model.eval()
    test_psnr, test_ssim = 0, 0
    for input_img, target_img, filename in tqdm(testloader):
        input_img, target_img = input_img.to(device), target_img.to(device)
        orgh, orgw = input_img.shape[2:]
        if opt.data_dir.split('/')[-3] in ['IJRR2017', 'CWFID', 'rice_s_n_w']:
            factor = 32
            h,w = input_img.shape[2], input_img.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = F.pad(input_img, (0,padw,0,padh), 'reflect')
        
        with torch.no_grad():
            restored = model(input_img)
        
        # [-1, 1] to [0, 1]
        restored_cpu, target_cpu = restored.detach().cpu(), target_img.detach().cpu()
        restored_cpu, target_cpu = list(map(lambda x: (x+1.0)/2.0, [restored_cpu, target_cpu]))[:]
        restored_cpu, target_cpu = restored_cpu[:, :, :orgh, :orgw], target_cpu[:, :, :orgh, :orgw]
        test_psnr += psnr(restored_cpu, target_cpu).item()
        test_ssim += ssim(restored_cpu, target_cpu).item()
            
        if opt.save_img:
            save_compare_img([target_cpu.numpy(), ((input_img.detach().cpu().numpy()[:, :, :orgh, :orgw])+1.0)/2.0, restored_cpu.numpy()], filename, save_dir)
            save_restored_img(restored_cpu.numpy(), filename, save_dir)
            
    test_psnr /= len(testloader)
    test_ssim /= len(testloader)
    
    testtxt = f"PSNR: {test_psnr:.4f} SSIM: {test_ssim:.4f}\n"
    print(testtxt)
    if opt.save_txt:
        f.write(testtxt)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../blur_cropweed/CWFID/MPR/test', help='directory that has data')
    parser.add_argument('--save_dir', type=str, default='./test/CWFID', help='directory for saving results')
    parser.add_argument('--weights', type=str, default='./train/CWFID/Full_FPWRN-ep20-10-colab-lr0.0001/ckpoints/best_ssim.pth', help='weights file for test')
    parser.add_argument('--save_img', type=bool, default=True, help='save result images')
    parser.add_argument('--save_txt', type=bool, default=True, help='save training process as txt file')
    parser.add_argument('--gpu', type=str, default='0', help='gpu number. -1 is cpu')
    
    opt = parser.parse_args()
    
    print(opt)
    model = SimpleNet(3)
    test(model, opt)