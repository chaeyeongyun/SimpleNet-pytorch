import math
import numpy as np
import torch
import torch.nn.functional as F


def psnr(restored_batch:torch.Tensor, target_batch:torch.Tensor, pixel_max=1.0):
    '''
    Returns the average psnr value of each image in batch
    Args:
        restored_batch(ndarray) : restored image batch (deblurred) that have pixel values in [0 1], shape:(N, 3, H, W)
        target_batch(ndarray) : target image batch (original) that have pixel values in [0 1], shape:(N, 3, H, W)
    Returns: psnr 
    '''
    mse = torch.mean((target_batch - restored_batch)**2) # mean of flattened array
    if mse == 0 : return float('inf') 
    psnr = 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr
    
    
def ssim(restored_batch:torch.Tensor, target_batch:torch.Tensor, windowsize=11, sigma=1.5, reduction='mean'):
    '''
    Args:
        pred_batch(torch.Tensor) : restored image batch (deblurred) that have pixel values in [0 1], shape:(N, 3, H, W)
        target_batch(torch.Tensor) : target image batch (original) that have pixel values in [0 1], shape:(N, 3, H, W)
    Returns: ssim 
    '''
    assert restored_batch.shape == target_batch.shape, "two Tensors' shapes must be same"
    channel = restored_batch.shape[1]
    C1, C2 = 0.01**2, 0.03**2
    C3 = C2/2
    ## gaussian kernel
    gaussian_window = torch.Tensor(
        [math.exp(-(x - windowsize//2)**2/float(2*sigma**2)) for x in range(windowsize)] # Normal distribution pdf formula
        ) # torch.Size([11])
    gaussian_window = gaussian_window / gaussian_window.sum()
    window_1D = gaussian_window.unsqueeze(1) # torch.Size([11,1])
    window_2D = window_1D.mm(window_1D.t()) # matrix multiplication (11,1) x (11, 1) -> (11, 11)
    window_2D = window_2D.float().unsqueeze(0).unsqueeze(0) # (11,11) -> (1, 11, 11) -> (1, 1, 11, 11)
    window = window_2D.expand(channel, 1, windowsize, windowsize).contiguous()
    
    if restored_batch.is_cuda: window=window.to(restored_batch.get_device())
    window = window.type_as(restored_batch)
    
    # mu : luminance
    mu_x = F.conv2d(restored_batch, window, padding=windowsize//2, groups=channel) # (N, C, H, W)
    mu_y = F.conv2d(target_batch, window, padding=windowsize//2, groups=channel) # (N, C, H, W)
    mu_xy = mu_x * mu_y # (N, C, H, W)
    
    mu_x_sq, mu_y_sq = mu_x.pow(2), mu_y.pow(2) # (N, C, H, W)
    # sigma : contrast
    sigma_x_sq = F.conv2d(restored_batch*restored_batch, window, padding=windowsize//2, groups=channel) - mu_x_sq # (N, C, H, W)
    sigma_y_sq = F.conv2d(target_batch*target_batch, window, padding=windowsize//2, groups=channel) - mu_y_sq # (N, C, H, W)
    sigma_xy = F.conv2d(restored_batch*target_batch, window, padding=windowsize//2, groups=channel) - mu_xy # (N, C, H, W)
    
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2))/((mu_x_sq+mu_y_sq+C1)*(sigma_x_sq + sigma_y_sq + C2)) # (N, C, H, W)
    
    # reduction
    if reduction == 'mean':
        return ssim_map.mean()
    elif reduction == 'sum':
        return ssim_map.sum()
    else:
        for i in range(len(restored_batch.shape)-1): ssim_map = ssim_map.mean(1)
        return ssim_map 
        
class SSIM(torch.nn.Module):
    def __init__(self, windowsize=11, sigma=1.5, reduction='mean') :
        super(SSIM, self).__init__()
        self.windowsize = windowsize
        self.reduction = reduction
        self.sigma = sigma
    
    def forward(self, restored_batch:torch.Tensor, target_batch:torch.Tensor):
        return ssim(restored_batch, target_batch, windowsize=self.windowsize, sigma=self.sigma, reduction=self.reduction)
    
    
    
if __name__ == '__main__':
    zerotoone = torch.rand((2, 3, 5, 5))
    monetoone = 2*(zerotoone-0.5)
    zeroto255 = (zerotoone*255).type(torch.int64)
    
    target_zerotoone = torch.rand((2, 3, 5, 5))
    target_monetoone = 2*(target_zerotoone-0.5)
    target_zeroto255 = (target_zerotoone*255).type(torch.int64)
    
    print(psnr(zerotoone, target_zerotoone))
    print(psnr(monetoone, target_monetoone))
    print(psnr(zeroto255.type(torch.float64), target_zeroto255.type(torch.float64), pixel_max=255.0))