import torch
from torchvision.transforms import Grayscale
from torchvision.utils import save_image


### video_transform functions, for converting raw video to PCA input ### 

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

def video_transform_5500_saveimage(video_tensor, video_name):
    grayscale = Grayscale(num_output_channels=1) # expected to have […, 3, H, W] shape
    # video_tensor.shape: torch.Size([124, 960, 540, 3])
    hmin = 900//2
    hmax = 1080//2
    wmin = 380//2
    wmax = 660//2
    
    print(video_tensor.shape)
    v = video_tensor[:,hmin:hmax, wmin:wmax,:,]
    v = torch.permute(v, (0,3,1,2)) # reshape to BCHW for grayscale
    v = grayscale(v)
    ## just for saving images
    images = torch.clone(v).to(torch.float64)
    images /= 255.0
    images = torch.index_select(images, dim=0, index=torch.Tensor([0,10,20,30]).to(torch.int32))
    save_image(images, f"results/imgs/{str(video_name)}_grid.png")
    ## end
    v = torch.squeeze(v, 1) # BCHW -> BHW
    B = v.shape[0]
    v = torch.reshape(v, (B, -1))  # BHW -> B(HW)
    
    return v

def video_transform_90x190(video_tensor):
    grayscale = Grayscale(num_output_channels=1) # expected to have […, 3, H, W] shape
    # video_tensor.shape: torch.Size([124, 960, 540, 3])
    hmin = 900//2
    hmax = 1080//2
    wmin = 380//2
    wmax = 660//2
    
    v = video_tensor[:,hmin:hmax, wmin:wmax,:,]
    v = torch.permute(v, (0,3,1,2)) # reshape to BCHW for grayscale
    v = grayscale(v)
    v = torch.squeeze(v, 1) # BCHW -> BHW
    B = v.shape[0]
    v = torch.reshape(v, (B, -1))  # BHW -> B(HW)
    
    return v

def video_transform_7797(video_tensor):
    grayscale = Grayscale(num_output_channels=1) # expected to have […, 3, H, W] shape
    # video_tensor.shape: torch.Size([124, 960, 540, 3])
    hmin = 890//2
    hmax = 1090//2
    wmin = 380//2
    wmax = 680//2
    
    v = video_tensor[:,hmin:hmax, wmin:wmax,:,]
    v = torch.permute(v, (0,3,1,2)) # reshape to BCHW for grayscale
    v = grayscale(v)
    v = torch.squeeze(v, 1) # BCHW -> BHW
    B = v.shape[0]
    v = torch.reshape(v, (B, -1))  # BHW -> B(HW)
    
    return v

### Frame transform functions and helpers, for concatenating audio and visual features for DPGMM input ###

def a2v_idx(audio_idx, video_fps):
    time_ms = audio_idx * 10 + 25/2 # offset some number between 0 and 25
    time_between_vidx = 1000 / video_fps
    video_idx = int(time_ms // time_between_vidx)
    return video_idx
    # first video frame should be at 0ms
    # check onsets of known "b"s or "p"s, make sure video frame is closed mouth

def frame_transform_5500(a_idx, video_features, audio_features, video_name_stem, video_fps):
    v_idx = a2v_idx(a_idx, video_fps)
    video_frame = video_features[v_idx]
    # add info from surrounding video frames to video features
    prev_video_frame = video_features[v_idx] if v_idx == 0 else video_features[v_idx-1]
    next_video_frame = video_features[v_idx] if v_idx == (video_features.shape[0]-1) else video_features[v_idx+1]
    # append audio slice + video slice to data
    frame_data = torch.cat(
        [audio_features[a_idx], 
            video_frame, 
            video_frame-prev_video_frame, 
            next_video_frame-video_frame, 
            torch.Tensor([int(video_name_stem)]), 
            torch.Tensor([a_idx])], 
            dim=-1,
    ) # torch.Size([178])
    return frame_data

def a2v_idx_offset(audio_idx, video_fps, offset):
    """ does not necessarily return an index >= 0 """
    time_ms = audio_idx * 10 + 25/2 - offset
    time_between_vidx = 1000 / video_fps
    video_idx = int(time_ms // time_between_vidx)
    # print(audio_idx, video_idx)
    return video_idx

def frame_transform_offset(a_idx, video_features, audio_features, video_name_stem, video_fps, offset):
    # if mode == 'a':
    #     return torch.cat(
    #         [audio_features[a_idx],
    #         torch.Tensor([int(video_name_stem)]), 
    #         torch.Tensor([a_idx])],
    #         dim=-1
    #     )
    v_idx = a2v_idx_offset(a_idx, video_fps, offset)
    if v_idx < 0:
        return None
    video_frame = video_features[v_idx]
    # add info from surrounding video frames to video features
    prev_video_frame = video_features[v_idx] if v_idx == 0 else video_features[v_idx-1]
    next_video_frame = video_features[v_idx] if v_idx == (video_features.shape[0]-1) else video_features[v_idx+1]
    # append audio slice + video slice to data
    frame_data = torch.cat(
        [audio_features[a_idx], 
            video_frame, 
            video_frame-prev_video_frame, 
            next_video_frame-video_frame, 
            torch.Tensor([int(video_name_stem)]), 
            torch.Tensor([a_idx])], 
            dim=-1,
    ) # torch.Size([178])
    return frame_data

frame_transform_offset_150 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset(a_idx, video_features, audio_features, video_name_stem, video_fps, 150)
frame_transform_offset_100 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset(a_idx, video_features, audio_features, video_name_stem, video_fps, 100)
frame_transform_offset_120 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset(a_idx, video_features, audio_features, video_name_stem, video_fps, 120)
frame_transform_offset_140 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset(a_idx, video_features, audio_features, video_name_stem, video_fps, 140)
frame_transform_offset_160 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset(a_idx, video_features, audio_features, video_name_stem, video_fps, 160)
frame_transform_offset_180 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset(a_idx, video_features, audio_features, video_name_stem, video_fps, 180)
frame_transform_offset_200 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset(a_idx, video_features, audio_features, video_name_stem, video_fps, 200)


def frame_transform_offset_frontpad(a_idx, video_features, audio_features, video_name_stem, video_fps, offset, frontpad=500):
    v_idx = a2v_idx_offset(int(a_idx+frontpad/10), video_fps, offset)
    if v_idx < 0:
        return None
    video_frame = video_features[v_idx]
    # add info from surrounding video frames to video features
    prev_video_frame = video_features[v_idx] if v_idx == 0 else video_features[v_idx-1]
    next_video_frame = video_features[v_idx] if v_idx == (video_features.shape[0]-1) else video_features[v_idx+1]
    # append audio slice + video slice to data
    frame_data = torch.cat(
        [audio_features[int(a_idx+frontpad/10)], 
            video_frame, 
            video_frame-prev_video_frame, 
            next_video_frame-video_frame, 
            torch.Tensor([int(video_name_stem)]), 
            torch.Tensor([a_idx])], 
            dim=-1,
    ) # torch.Size([178])
    return frame_data

frame_transform_offset_frontpad_110 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset_frontpad(a_idx, video_features, audio_features, video_name_stem, video_fps, 110)
frame_transform_offset_frontpad_120 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset_frontpad(a_idx, video_features, audio_features, video_name_stem, video_fps, 120)
frame_transform_offset_frontpad_130 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset_frontpad(a_idx, video_features, audio_features, video_name_stem, video_fps, 130)
frame_transform_offset_frontpad_140 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset_frontpad(a_idx, video_features, audio_features, video_name_stem, video_fps, 140)

def frame_transform_offset_frontpad_nocontext(a_idx, video_features, audio_features, video_name_stem, video_fps, offset, frontpad=500):
    v_idx = a2v_idx_offset(int(a_idx+frontpad/10), video_fps, offset)
    if v_idx < 0:
        return None
    video_frame = video_features[v_idx]
    # append audio slice + video slice to data
    frame_data = torch.cat(
        [audio_features[int(a_idx+frontpad/10)], 
            video_frame, 
            torch.Tensor([int(video_name_stem)]), 
            torch.Tensor([a_idx])], 
            dim=-1,
    )
    return frame_data

frame_transform_offset_frontpad_nocontext_120 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset_frontpad_nocontext(a_idx, video_features, audio_features, video_name_stem, video_fps, 120)


def frame_transform_offset_frontpad_dd(a_idx, video_features, audio_features, video_name_stem, video_fps, offset, frontpad=500):
    v_idx = a2v_idx_offset(int(a_idx+frontpad/10), video_fps, offset)
    if v_idx < 0:
        return None
    video_frame = video_features[v_idx]
    # add info from surrounding video frames to video features
    prev_video_frame = video_features[v_idx] if v_idx == 0 else video_features[v_idx-1]
    next_video_frame = video_features[v_idx] if v_idx == (video_features.shape[0]-1) else video_features[v_idx+1]
    # append audio slice + video slice to data
    frame_data = torch.cat(
        [audio_features[int(a_idx+frontpad/10)], 
            video_frame, 
            video_frame-prev_video_frame, 
            next_video_frame-video_frame, 
            (next_video_frame-video_frame) - (video_frame-prev_video_frame),
            torch.Tensor([int(video_name_stem)]), 
            torch.Tensor([a_idx])], 
            dim=-1,
    ) # torch.Size([178])
    return frame_data

frame_transform_offset_frontpad_dd_120 = lambda a_idx, video_features, audio_features, video_name_stem, video_fps: frame_transform_offset_frontpad_dd(a_idx, video_features, audio_features, video_name_stem, video_fps, 120)
