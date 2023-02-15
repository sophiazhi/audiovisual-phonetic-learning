"""
using pretrained visual features PCA model,
extract audio-visual features from .mp4 videos
"""

import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.io import savemat
from joblib import dump, load
import json
import torchaudio
from torchvision.io import read_video
import torch
import os
from video_library import video_transform_5500_230212 as video_transform
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for audiovisual feature extraction')
    
    parser.add_argument('train_json', help='Path to json list of training videos')
    parser.add_argument('model_name', help='Path to video model pkl file')
    parser.add_argument('output_file', help='Path to output .mat or .npy file to write features to')
    parser.add_argument('--video_corpus', default='../corpora/childes_synthetic_video_60fps/')
    parser.add_argument('--audio_corpus', default='../corpora/childes_synthetic_audio/')
    parser.add_argument('--split', default='train') # can also be 'test' or 'eval'

    args = parser.parse_args()
    return args


def read_train_list(train_json):
    with open(train_json, 'r') as f:
        train_list = json.load(f)
    return train_list


def a2v_idx(audio_idx, video_fps):
    time_ms = audio_idx * 10 + 25/2 # offset some number between 0 and 25
    time_between_vidx = 1000 / video_fps
    video_idx = int(time_ms // time_between_vidx)
    return video_idx


def process_videos(train_list, pca_model, video_corpus, audio_corpus):
    audio_sample_rate = read_video(os.path.splitext(os.path.join(video_corpus, train_list[0]))[0] + ".mp4", pts_unit='sec')[2]['audio_fps']

    # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MFCC.html#torchaudio.transforms.MFCC
    # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html#torchaudio.transforms.MelSpectrogram
    mfcc_transform = torchaudio.transforms.MFCC(
        log_mels = True,
        sample_rate=audio_sample_rate, 
        n_mfcc=26,
        melkwargs={
            # "sample_rate": audio_sample_rate,
            "n_fft": int(audio_sample_rate * 25/1000.),
            "n_mels": 64,
            # "win_length": 25,
            "hop_length": int(audio_sample_rate * 10/1000.),
        },
    )
    
    data = []
    for video_name_idx, video_name in enumerate(tqdm(train_list)):
        video_info = read_video(os.path.splitext(os.path.join(video_corpus, video_name))[0] + ".mp4", pts_unit='sec')
        video_tensor = video_info[0] # BHWC, torch.Size([742, 1080, 1920, 3])
        video_features = video_transform(video_tensor)
        video_features = pca_model.transform(video_features) # 60fps
        video_features = torch.Tensor(video_features)
        # print(video_features.shape) # Bx50
        
        audio_tensor = video_info[1] # torch.Size([1, frames])
        audio_features = mfcc_transform(audio_tensor) # torch.Size([1, 26, 17613])
        audio_features = torch.transpose(torch.squeeze(audio_features, dim=0), 0, 1)
        # print(audio_features.shape) # torch.Size([17613, 26])
        
        # check length of original unpadded audio so we don't include the silence at the end
        original_audio, sr = torchaudio.load(os.path.splitext(os.path.join(audio_corpus, video_name))[0] + ".wav")
        num_audio_feature_frames = (original_audio.shape[1] - int(sr * 25/1000.))//int(sr * 10/1000.) + 1
        # print(num_audio_feature_frames)
        
        # match video frames to audio frames
        for a_idx in range(num_audio_feature_frames):
            v_idx = a2v_idx(a_idx, video_info[2]['video_fps'])
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
                 torch.Tensor([int(Path(video_name).stem)]), # os.path.splitext(video_name)[0].split('/')[1] # torch.Tensor([video_name_idx]), 
                 torch.Tensor([a_idx])], 
                 dim=-1,
            ) # torch.Size([178])
            data.append(torch.unsqueeze(frame_data, dim=0)) # torch.Size([1, 178])
    
    data_tensor = torch.cat(data, dim=0)
    data_ndarray = data_tensor.detach().cpu().numpy()
    print(data_ndarray.shape)
    return data_ndarray


def np2out(data, output_file, split='train'):
    if split == 'train':
        np2mat(data, output_file)
    elif split == 'eval' or split == 'test':
        np2npy(data, output_file)

def np2mat(data, output_file):
    data = data.astype('float64')
    train_list = {}
    for i in tqdm(range(data.shape[0])):
        row = data[i,:]
        label = f"file{int(row[-2])}_window{int(row[-1])}"
        train_list[label] = row[:-2]
    savemat(output_file, train_list)

def np2npy(data, output_file):
    np.save(output_file, data)


if __name__ == '__main__':
    args = parse_args()
    
    train_list = read_train_list(args.train_json)
    pca_model = load(args.model_name)
    train_data = process_videos(train_list, pca_model, args.video_corpus, args.audio_corpus)
    np2out(train_data, args.output_file, args.split)
