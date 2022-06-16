from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

## Basic Conv Block : conv2d - batchnorm - act
class BasicConv(nn.Sequential):
    """
    Basic Conv Block : conv2d - batchnorm - act
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size, stride:int=1, padding:int=0, dilation=1, groups=1, bias=True, batch_norm=True, act:nn.Module=nn.ReLU()):
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
        if batch_norm:
            modules += [nn.BatchNorm2d(out_channels)]
        if act is not None:
            modules += [act]
        super().__init__(*modules)

## (original) ResBlock : { (Conv2d - BN - ReLu) - Conv2d - BN } - sum - ReLU
class ResidualBlock(nn.Module):
    """
    ResBlock : { (Conv2d - BN - ReLu) - Conv2d - BN } - sum - ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.convs = nn.Sequential(
            BasicConv(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True, batch_norm=True),
            BasicConv(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True, batch_norm=True, act=None)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        conv_out = self.convs(x)
        output = x + conv_out
        output = self.relu(output)
        return output
    
### Deformable Residual Blocks
class DeformableConv2d(nn.Module):
    '''Deformable convolution block'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, dilation=1):
        super(DeformableConv2d, self).__init__()
        assert type(kernel_size) in (int, tuple), "type of kernel_size must be int or tuple"
        kernel_size = (kernel_size, kernel_size) if type(kernel_size)==int else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # conv layer to calculate offset 
        self.offset_conv = nn.Conv2d(in_channels, 2*kernel_size[0]*kernel_size[1], kernel_size=kernel_size, stride=stride, padding=(kernel_size[0]-1)//2, bias=True)
        # conv layer to calculate modulator
        self.modulator_conv = nn.Conv2d(in_channels, kernel_size[0]*kernel_size[1], kernel_size=kernel_size, stride=stride, padding=(kernel_size[0]-1)//2, bias=True)
        # conv layers for offset and modulator must be initilaized to zero.
        self.zero_init([self.offset_conv, self.modulator_conv])
        
        # conv layer for deformable conv. offset and modulator will be adapted to this layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        
    def zero_init(self, convlayer):
        if type(convlayer) == list:
            for c in convlayer:
                nn.init.constant_(c.weight, 0.)
                nn.init.constant_(c.bias, 0.)
        else:
            nn.init.constant_(convlayer.weight, 0.)
            nn.init.constant_(convlayer.bias, 0.)
    
    def forward(self, x):
        offset = self.offset_conv(x)
        modulator =  torch.sigmoid(self.modulator_conv(x)) # modulator has (0, 1) values.
        output = torchvision.ops.deform_conv2d(input=x, 
                               offset=offset,
                               weight=self.conv.weight,
                               bias=self.conv.bias,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=self.dilation,
                               mask=modulator)
        return output

class Deformable_Resblock(nn.Module):
    def __init__(self, in_channels, deformable_out_channels:int, kernel_size, stride:int=1, padding:int=0, dilation=1, bias=True, batch_norm=True, act:nn.Module=nn.ReLU()):
        super().__init__()
        self.convs = nn.Sequential(DeformableConv2d(in_channels, deformable_out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
                                   act)
        self.last_conv = nn.Conv2d(deformable_out_channels, in_channels, kernel_size=3, padding=1, bias=True)
    
    def forward(self, x):
        convs_out = self.convs(x)
        last_conv_out = self.last_conv(convs_out)
        return x + last_conv_out
    
## Attention Layers
class CALayer(nn.Module):
    def __init__(self, in_channels, intermidiate_channels, act=nn.PReLU(), only_GAP=True):
        super().__init__()
        if only_GAP:
            self.pooling = nn.ModuleList([nn.AdaptiveAvgPool2d(1)])
        else:
            self.pooling = nn.ModuleList([nn.AdaptiveMaxPool2d(1), nn.AdaptiveAvgPool2d(1)])
        
        self.shared_1x1_conv = nn.Sequential(nn.Conv2d(in_channels, intermidiate_channels, 1, padding=0, bias=True), 
                                             act, 
                                             nn.Conv2d(intermidiate_channels, in_channels, 1, padding=0, bias=True),
                                             )
    def forward(self, x):
        pooling = [self.shared_1x1_conv(pool(x)) for pool in self.pooling]
        pooling_sum = pooling[0] if len(pooling)==1 else pooling[0]+pooling[1]
        ca = torch.sigmoid(pooling_sum)
        output = x * ca
        return output


## PVB
class PVB(nn.Module):
    def __init__(self, in_channels, atrous_rates:List[int]=[1, 2, 3], act=nn.ReLU()):
        super().__init__()
        modules = []
        for rate in atrous_rates:
            modules += [BasicConv(in_channels, in_channels//2, kernel_size=3, padding=rate, batch_norm=True, bias=False, act=act, dilation=rate)]
        modules += [nn.Sequential(DeformableConv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(in_channels//2),
                                  act)]
        self.convlist = nn.ModuleList(modules)
        self.project = DeformableConv2d(len(self.convlist)*in_channels//2, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        conv_results = []
        for conv in self.convlist:
            conv_results += [conv(x)]
        conv_output = torch.cat(tuple(conv_results), dim=1)
        conv_output = self.project(conv_output)
        output = x + conv_output
        return output

## LGCR : Light Global Context Refinement
class LGCR(nn.Module):
    def __init__(self, cp_rank=64):
        super().__init__()
        # cp_rank is r in paper
        self.conv1d_c = nn.Sequential(nn.Conv1d(1, cp_rank, 3, padding=1, bias=False),
                                      nn.PReLU())
        self.conv1d_w = nn.Sequential(nn.Conv1d(1, cp_rank, 3, padding=1, bias=False),
                                      nn.PReLU())
        self.conv1d_h = nn.Sequential(nn.Conv1d(1, cp_rank, 3, padding=1, bias=False),
                                      nn.PReLU())
        self.conv3d = nn.Conv3d(1, 1, 3, padding=1, bias=False)
    
    def forward(self, x):
        # x.shape = (N, C, H, W)
        xshape = x.shape
        ### CPtensor reconstuction
        #GAP_C
        gap_c = (F.adaptive_avg_pool2d(x, 1)).squeeze(dim=-1) # (N, C, 1, 1) (N, C, 1)
        c_out = torch.swapaxes(self.conv1d_c(torch.swapaxes(gap_c, -1, -2)), -1, -2) # (N, C, 1)->(N, 1, C)->(N, 64, C)->(N, C, r)
        sp_c_out = torch.split(c_out, 1, dim=-1) # [(N, C, 1) x r]
        #GAP_W
        gap_w = torch.mean(torch.mean(x, dim=1, keepdim=True), dim=2) # (N, 1, W)
        w_out = self.conv1d_w(gap_w) # (N, r, W)
        #GAP_H
        gap_h = torch.mean(torch.mean(x, dim=1, keepdim=True), dim=3) # (N, 1, H)
        h_out = torch.swapaxes(self.conv1d_h(gap_h), -1, -2) # (N, r, H) (N, H, r)
        sp_h_out, sp_w_out = torch.split(h_out, 1, dim=-1), torch.split(w_out, 1, dim=1) # [(N, H, 1)xr] [(N, 1, W)xr]
        
        l = [torch.einsum('bij, bjk -> bik', sp_h, sp_w).unsqueeze(1) for sp_h, sp_w in zip(sp_h_out, sp_w_out)] # [(N, 1, H, W) x r]
        HW64 = torch.cat(tuple(l), dim=1) # (N, r, H, W)
        HW64 = HW64.view((HW64.shape[0], HW64.shape[1], HW64.shape[2]*HW64.shape[3])) # (N, r, HxW)
        sp_HW64 = torch.split(HW64, 1, dim=1) # [(N, 1, HxW) x r]
        l = [torch.einsum('bij, bjk -> bik', sp_c, sp_hw).view(xshape).unsqueeze(dim=0) for sp_c, sp_hw in zip(sp_c_out, sp_HW64)] # (N, C, HxW) ->(N, C, H, W)->(1, N, C, H, W)x64
        cat = torch.cat(tuple(l), dim=0) # (r, N, C, H, W)
        cp_out = torch.sum(cat, dim=0) # (N, C, H, W)
        GCRW = F.softmax(cp_out, dim=1)
        
        ## Raw Residual
        raw_residual = self.conv3d(x.unsqueeze(dim=1)) # (N, 1, C, H, W) -> (N, 1, C, H, W)
        raw_residual = raw_residual.squeeze(dim=1) # (N, C, H, W)
        
        output = torch.mul(raw_residual, GCRW) + x
        return output
    
## LGCR : Light Global Context Refinement
class LGCR_ver2(nn.Module):
    def __init__(self, cp_rank=64):
        super().__init__()
        # cp_rank is r in paper
        self.conv1d_c = nn.Sequential(nn.Conv1d(1, cp_rank, 3, padding=1, bias=False),
                                      nn.PReLU())
        self.conv1d_w = nn.Sequential(nn.Conv1d(1, cp_rank, 3, padding=1, bias=False),
                                      nn.PReLU())
        self.conv1d_h = nn.Sequential(nn.Conv1d(1, cp_rank, 3, padding=1, bias=False),
                                      nn.PReLU())
        self.conv3d = nn.Conv3d(1, 1, 3, padding=1, bias=False)
    
    def forward(self, x):
        # x.shape = (N, C, H, W)
        xshape = x.shape
        ### CPtensor reconstuction
        #GAP_C
        gap_c = (F.adaptive_avg_pool2d(x, 1)).squeeze(dim=-1) # (N, C, 1, 1) (N, C, 1)
        c_out = torch.swapaxes(self.conv1d_c(torch.swapaxes(gap_c, -1, -2)), -1, -2) # (N, C, 1)->(N, 1, C)->(N, 64, C)->(N, C, r)
        
        #GAP_W
        gap_w = torch.mean(torch.mean(x, dim=1, keepdim=True), dim=2) # (N, 1, W)
        w_out = self.conv1d_w(gap_w) # (N, r, W)
        w_out = w_out.unsqueeze(dim=-2) # (N, r, 1, W)
        #GAP_H
        gap_h = torch.mean(torch.mean(x, dim=1, keepdim=True), dim=3) # (N, 1, H)
        h_out = self.conv1d_h(gap_h) # (N, r, H) 
        h_out = h_out.unsqueeze(dim=-1) # (N, r, H, 1)
        
        nrhw = torch.einsum('bchi, bciw -> bchw', h_out, w_out) # (N, r, H, W)
        # nrhw = torch.flatten(nrhw, start_dim=-2) # (N, r, HxW)
        nchw = torch.einsum('bcr, brhw -> brchw', c_out, nrhw) # (N, r, C, H, W)
        nchw = torch.sum(nchw, dim=1) # (N, C, H, W)
        GCRW = F.softmax(nchw, dim=1)
        
        ## Raw Residual
        raw_residual = self.conv3d(x.unsqueeze(dim=1)) # (N, 1, C, H, W) -> (N, 1, C, H, W)
        raw_residual = raw_residual.squeeze(dim=1) # (N, C, H, W)
        
        output = torch.mul(raw_residual, GCRW) + x
        return output    

## Encoder
class EncoderBlock(nn.Sequential):
    def __init__(self, in_channels):
        modules = [ResidualBlock(in_channels, in_channels),
                   PVB(in_channels)]
        super().__init__(*modules)
        
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.vanilla_conv = BasicConv(in_channels, out_channels, 3, padding=1, bias=True, batch_norm=True)
        self.enc1 = EncoderBlock(out_channels)
        self.down1 = nn.UpsamplingBilinear2d(scale_factor=0.5)
        self.enc2 =  EncoderBlock(out_channels)
        self.down2 = nn.UpsamplingBilinear2d(scale_factor=0.5)
        self.enc3 = EncoderBlock(out_channels)
    def forward(self, x):
        conv_out = self.vanilla_conv(x) # (N, 256, H, W)
        skip1 = self.enc1(conv_out)
        skip2 = self.enc2(self.down1(skip1))
        enc_out = self.enc3(self.down2(skip2))
        return enc_out, skip1, skip2

## Decoder
class DecoderBlock(nn.Sequential):
    def __init__(self, in_channels):
        modules = [ResidualBlock(in_channels, in_channels),
                   PVB(in_channels),
                   Deformable_Resblock(in_channels, in_channels, kernel_size=3, padding=1, bias=True, batch_norm=True)]
        super().__init__(*modules)

class Decoder(nn.Module):
    def __init__(self, enc_out_channels):
        super().__init__()
        self.dec_block_1 = DecoderBlock(enc_out_channels)
        self.up1 = nn.ConvTranspose2d(enc_out_channels, enc_out_channels, kernel_size=4, stride=2, padding=1)
        
        self.dec_block_2 = DecoderBlock(enc_out_channels*2)
        self.up2 = nn.ConvTranspose2d(enc_out_channels*2, enc_out_channels, kernel_size=4, stride=2, padding=1)
        
        self.dec_block_3 = DecoderBlock(enc_out_channels*2)

        self.last_conv = nn.Sequential(BasicConv(enc_out_channels*2, 3, kernel_size=3, padding=1, bias=True, batch_norm=False))
    
    def forward(self, lgcr_out, skip1, skip2):
        dec1 = self.dec_block_1(lgcr_out)
        up1 = self.up1(dec1) # (N, C, H/2, W/2)
        dec2 = self.dec_block_2(torch.cat((skip2, up1), dim=1))
        up2 = self.up2(dec2) # (N, C, H, W)
        dec3 = self.dec_block_3(torch.cat((skip1, up2), dim=1))
        output = self.last_conv(dec3) # (N, 3, H, W)
        return output

## SimpleNet
class SimpleNet(nn.Module):
    def __init__(self, in_channels, enc_out_channels=256, cp_rank=64):
        super().__init__()
        self.encoder = Encoder(in_channels, enc_out_channels)
        self.lgcr = LGCR(cp_rank)
        self.decoder = Decoder(enc_out_channels)
    
    def forward(self, x):
        enc_out, skip1, skip2 = self.encoder(x)
        lgcr_out = self.lgcr(enc_out)
        dec_out = self.decoder(lgcr_out, skip1, skip2)
        output = dec_out + x
        output = torch.tanh(output)
        return output
        

