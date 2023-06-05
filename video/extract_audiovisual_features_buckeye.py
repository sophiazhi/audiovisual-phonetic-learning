"""
using pretrained visual features PCA model,
extract audio-visual features from .mp4 videos
"""

import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.io import savemat
from joblib import dump, load
import pickle
import json
import torchaudio
from torchvision.io import read_video
import torch
import os
import video_library # from video_library import video_transform_90x190 as video_transform, frame_transform_5500 as frame_transform
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for audiovisual feature extraction')
    
    parser.add_argument('train_json', help='Path to json list of training videos')
    parser.add_argument('model_name', help='Path to video model pkl file')
    parser.add_argument('output_file', help='Path to output .mat or .npy file to write features to')
    parser.add_argument('--video_fn', default='video_transform_7797', help='video transform function name')
    parser.add_argument('--frame_fn', default='frame_transform_5500', help='frame transform function name')
    parser.add_argument('--video_corpus', default='/om2/user/szhi/corpora/buckeye_synthetic_video/')
    parser.add_argument('--audio_corpus', default='/om2/user/szhi/corpora/buckeye_segments/')
    parser.add_argument('--split', default='train') # can also be 'test' or 'eval'
    parser.add_argument('--mode', default='av', help='Modality/ies of features to extract. Can be "a" (audio only), "v", "n" (noisy audio), "av", or "nv".')

    args = parser.parse_args()
    return args


def read_train_list(train_json):
    with open(train_json, 'r') as f:
        train_list = json.load(f)
    return train_list


def process_videos(
    train_list, 
    pca_model, 
    video_transform, 
    frame_transform, 
    video_corpus, 
    audio_corpus, 
    split, 
    output_file,
    mode='av',
    ):
    train = (split == 'train')
    
    audio_sample_rate = read_video(os.path.splitext(os.path.join(video_corpus, train_list[0]))[0] + ".mp4", pts_unit='sec')[2]['audio_fps']

    # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MFCC.html#torchaudio.transforms.MFCC
    # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html#torchaudio.transforms.MelSpectrogram
    mfcc_transform = torchaudio.transforms.MFCC( # "be1375fy_043023"
        log_mels = True,
        sample_rate=audio_sample_rate, 
        n_mfcc=13,
        melkwargs={
            "n_fft": int(audio_sample_rate * 25/1000.),
            "n_mels": 23,
            "hop_length": int(audio_sample_rate * 10/1000.),
            "center": False,
        },
    )
    delta_transform = torchaudio.transforms.ComputeDeltas(win_length=7)

    if train:
        train_feats = {}
    else:
        features = []
        utt_ids = []
        times = []
    
    for video_name_idx, video_name in enumerate(tqdm(train_list)):
        if not os.path.exists(os.path.splitext(os.path.join(video_corpus, video_name))[0] + ".mp4"):
            print(f"Missing file {os.path.splitext(video_name)[0]}")
            continue

        video_info = read_video(os.path.splitext(os.path.join(video_corpus, video_name))[0] + ".mp4", pts_unit='sec')
        video_tensor = video_info[0] # BHWC, torch.Size([742, 1080, 1920, 3])
        video_features = video_transform(video_tensor)
        video_features = pca_model.transform(video_features) # 60fps
        video_features = torch.Tensor(video_features)
        # print(video_features.shape) # Bx50
        video_fps = video_info[2]['video_fps']
        
        audio_tensor = video_info[1] # torch.Size([1, frames])
        if mode in set('n', 'nv'):
            noise = torch.randn(audio_tensor.shape) # torch.Size([1, frames])
            snr = torch.Tensor([5])
            audio_tensor = video_library.add_noise(audio_tensor, noise, snr) # torch.Size([1, frames])
        mfcc_features = mfcc_transform(audio_tensor) # torch.Size([1, 13, 17613])
        d_mfcc_features = delta_transform(mfcc_features)
        dd_mfcc_features = delta_transform(d_mfcc_features)
        audio_features = torch.cat((mfcc_features, d_mfcc_features, dd_mfcc_features), dim=1)
        audio_features = torch.transpose(torch.squeeze(audio_features, dim=0), 0, 1)
        # print(audio_features.shape) # torch.Size([17613, 39])
        
        # check length of original unpadded audio so we don't include the silence at the end
        original_audio, sr = torchaudio.load(os.path.splitext(os.path.join(audio_corpus, video_name))[0] + ".wav")
        num_audio_feature_frames = (original_audio.shape[1] - int(sr * 25/1000.))//int(sr * 10/1000.) + 1

        if len(video_features) < (num_audio_feature_frames*10+25 + 500) // (1000/video_fps):
            print(f"{os.path.splitext(video_name)[0]} missing video frames")
            print(len(audio_features), num_audio_feature_frames)
            print(len(video_features))
            continue
        
        if not train:
            data = []
        # match video frames to audio frames
        for a_idx in range(num_audio_feature_frames):
            # ## NEW FOR BUCKEYE
            # video_name_stem = Path(video_name).stem
            # video_name_stem = video_name_stem[3:5] + ('1' if video_name_stem[5] == 'a' else '2') + video_name_stem[7:]
            # ## 
            video_name_stem = video_library.vname2id_buckeye(Path(video_name).stem)
            frame_data = frame_transform(a_idx, video_features, audio_features, video_name_stem, video_fps)
            if frame_data is None: # video frame is out of range
                continue

            if mode == 'a' or mode == 'n':
                frame_data[audio_features.shape[1]] = frame_data[-2]
                frame_data[audio_features.shape[1]+1] = frame_data[-1]
                frame_data = frame_data[:audio_features.shape[1]+2]
            elif mode == 'v':
                frame_data = frame_data[audio_features.shape[1]:]
            # if mode == 'av' or 'nv' change nothing

            if train:
                frame_data = frame_data.detach().cpu().numpy().astype('float64')
                label = f"file{int(frame_data[-2])}_window{int(frame_data[-1])}"
                train_feats[label] = frame_data[:-2]
            else:
                data.append(torch.unsqueeze(frame_data, dim=0)) # torch.Size([1, 178])
        
        if not train:
            data_tensor = torch.cat(data, dim=0)
            data_ndarray = data_tensor.detach().cpu().numpy().astype('float64')
            features.append(data_ndarray[:, :-2])
            utt_ids.append(Path(video_name).stem)
            times.append(data_ndarray[:, -1] * 0.010 + 0.0125)

    if train:
        savemat(output_file, train_feats)
    else:
        sorted_zip = sorted(zip(utt_ids, times, features), key=lambda triple: triple[0])
        utt_ids = [str(u) for u,_,_ in sorted_zip]
        times = [t for _,t,_ in sorted_zip]
        features = [f for _,_,f in sorted_zip]

        with open(output_file, 'wb') as f:
            pickle.dump((utt_ids, times, features), f, protocol=2)


if __name__ == '__main__':
    args = parse_args()
    
    train_list = read_train_list(args.train_json)
    
    pca_model = load(args.model_name)
    video_transform = getattr(video_library, args.video_fn)
    frame_transform = getattr(video_library, args.frame_fn)
    
    process_videos(train_list, pca_model, video_transform, frame_transform, args.video_corpus, args.audio_corpus, args.split, args.output_file, args.mode, args.means_npy)
