import os
import numpy as np
import random
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def change_pixel_range(x:torch.Tensor):
    """chanhe pixel range to [-1, 1] from [0, 1]

    Args:
        x (torch.Tensor): input torch Tensor

    Returns:
        y : torch Tensor that have value in range[ -1, 1 ]
    """
    y = 2*(x-0.5)
    return y

class CropDataset(Dataset):
    ''' for train '''
    def __init__(self, data_dir, cropsize, randomaug=True):
        input_images = sorted(os.listdir(os.path.join(data_dir, 'input'))) # ['rgb_00254_2.png', 'rgb_00094_2.png', 'rgb_00185.png', ... ]
        target_images = sorted(os.listdir(os.path.join(data_dir, 'target')))
        assert len(input_images) == len(target_images), "the number of input images and output images must be the same"
        
        self.input_path = [os.path.join(data_dir, 'input', x) for x in input_images]
        self.target_path = [os.path.join(data_dir, 'target', x) for x in target_images]
        
        self.cropsize = cropsize
        self.randomaug = randomaug
        
    def __len__(self):
        return len(self.input_path)
    
    def __getitem__(self, idx): 
        cropsize = self.cropsize
        assert self.input_path[idx].split('/')[-1] == self.target_path[idx].split('/')[-1], \
                    "input image and target image are not matched"
        input_img = Image.open(self.input_path[idx])
        target_img = Image.open(self.target_path[idx])
        filename = self.input_path[idx].split('/')[-1] 
        if input_img.size != target_img.size: 
            target_img = target_img.resize(input_img.size)
            
        assert all([x >= cropsize for x in input_img.size]), "cropsize must smaller than image size"
        # TODO : random augmentation
        ## MPRNet 방법도 좋지만 비율을 정하는게 좋을 것 같다
        # 일단 없이 해보기
        # if self.randomaug:  
        input_img = TF.to_tensor(input_img)
        target_img = TF.to_tensor(target_img)
        
        # crop patch
        random.seed(1)
        x1 = random.randint(0, input_img.size()[2]-cropsize)
        y1 = random.randint(0, input_img.size()[1]-cropsize)
        input_img = input_img[:, y1:y1+cropsize, x1:x1+cropsize]
        target_img = target_img[:, y1:y1+cropsize, x1:x1+cropsize]
        
        input_img, target_img = change_pixel_range(input_img), change_pixel_range(target_img) # Tensor that have values in range [-1, 1]
        return input_img, target_img, filename
        

        
class FullDataset(Dataset):
    ''' for test '''
    def __init__(self, data_dir):
        input_images = sorted(os.listdir(os.path.join(data_dir, 'input'))) # ['rgb_00254_2.png', 'rgb_00094_2.png', 'rgb_00185.png', ... ]
        target_images = sorted(os.listdir(os.path.join(data_dir, 'target')))
        assert len(input_images) == len(target_images), "the number of input images and output images must be the same"
        
        self.input_path = [os.path.join(data_dir, 'input', x) for x in input_images]
        self.target_path = [os.path.join(data_dir, 'target', x) for x in target_images]
        
        
    def __len__(self):
        return len(self.input_path)
    
    def __getitem__(self, idx):
        assert self.input_path[idx].split('/')[-1] == self.target_path[idx].split('/')[-1], \
                    "input image and target image are not matched"
                    
        filename = self.input_path[idx].split('/')[-1] 
        input_img = Image.open(self.input_path[idx])
        target_img = Image.open(self.target_path[idx])
        
        input_img = TF.to_tensor(input_img)
        target_img = TF.to_tensor(target_img)
        
        input_img, target_img = change_pixel_range(input_img), change_pixel_range(target_img) # Tensor that have values in range [-1, 1]
        return input_img, target_img, filename
        
