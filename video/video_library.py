import torch
from torchvision.transforms import Grayscale

def video_transform_5500(video_tensor):
    grayscale = Grayscale(num_output_channels=1) # expected to have […, 3, H, W] shape
    # hmin, hmax = 475, 575
    # wmin, wmax = 175, 375
    hmin = 900
    hmax = 1080
    wmin = 380
    wmax = 660
    
    v = video_tensor[:,hmin:hmax, wmin:wmax,:,]
    v = torch.permute(v, (0,3,1,2)) # reshape to BCHW for grayscale
    v = grayscale(v)
    v = torch.squeeze(v, 1) # BCHW -> BHW
    B = v.shape[0]
    v = torch.reshape(v, (B, -1))  # BHW -> B(HW)
    
    return v

def video_transform_5500_230212(video_tensor):
    grayscale = Grayscale(num_output_channels=1) # expected to have […, 3, H, W] shape
    # hmin, hmax = 475, 575
    # wmin, wmax = 175, 375
    hmin = 900
    hmax = 1080
    wmin = 380
    wmax = 660
    
    v = video_tensor[:,hmin:hmax, wmin:wmax,:,] # BHWC
    v = torch.permute(v, (0,3,1,2)) # reshape to BCHW for grayscale
    v = torch.nn.functional.interpolate(v, (45, 70)) # downsample to fewer pixels
    v = grayscale(v)
    v = torch.squeeze(v, 1) # BCHW -> BHW
    B = v.shape[0]
    v = torch.reshape(v, (B, -1))  # BHW -> B(HW)
    
    return v