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
    parser.add_argument('--mode', default='av', help='Modality/ies of features to extract. Can be "a" (audio only), "v", "an" (audio with noise for visual), "vn", or "av".')
    parser.add_argument('--means_npy', default=None, help='Path to .npy file of pretrain feature means')

    args = parser.parse_args()
    return args


def read_train_list(train_json):
    with open(train_json, 'r') as f:
        train_list = json.load(f)
    print(len(train_list))
    return train_list

def add_noise(waveform, noise, snr):
    if not (waveform.ndim - 1 == noise.ndim - 1 == snr.ndim):
        raise ValueError("Input leading dimensions don't match.")

    L = waveform.size(-1)

    if L != noise.size(-1):
        raise ValueError(f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)}).")

    masked_waveform = waveform
    masked_noise = noise

    energy_signal = torch.linalg.vector_norm(masked_waveform, ord=2, dim=-1) ** 2  # (*,)
    energy_noise = torch.linalg.vector_norm(masked_noise, ord=2, dim=-1) ** 2  # (*,)
    original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
    scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

    # scale noise
    scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)

    return waveform + scaled_noise  # (*, L)


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
    means_npy=None,
    ):
    if split == 'train':
        train = True
    elif split == 'test' or split == 'eval':
        train = False
    
    audio_sample_rate = read_video(os.path.splitext(os.path.join(video_corpus, train_list[0]))[0] + ".mp4", pts_unit='sec')[2]['audio_fps']

    # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MFCC.html#torchaudio.transforms.MFCC
    # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html#torchaudio.transforms.MelSpectrogram
    # mfcc_transform = torchaudio.transforms.MFCC(
    #     log_mels = True,
    #     sample_rate=audio_sample_rate, 
    #     n_mfcc=13,
    #     melkwargs={
    #         # "sample_rate": audio_sample_rate,
    #         "n_fft": int(audio_sample_rate * 25/1000.),
    #         "n_mels": 64,
    #         # "win_length": 25,
    #         "hop_length": int(audio_sample_rate * 10/1000.),
    #     },
    # )
    # mfcc_transform = torchaudio.transforms.MFCC( # "be1375fy_vdim4_mfccdd39_042823"
    #     log_mels = True,
    #     sample_rate=audio_sample_rate, 
    #     n_mfcc=13,
    #     melkwargs={
    #         # "sample_rate": audio_sample_rate,
    #         "n_fft": int(audio_sample_rate * 25/1000.),
    #         "n_mels": 64,
    #         # "win_length": 25,
    #         "hop_length": int(audio_sample_rate * 10/1000.),
    #         "center": False,
    #     },
    # )
    mfcc_transform = torchaudio.transforms.MFCC( # "be1375fy_043023"
        log_mels = True,
        sample_rate=audio_sample_rate, 
        n_mfcc=13,
        melkwargs={
            # "sample_rate": audio_sample_rate,
            "n_fft": int(audio_sample_rate * 25/1000.),
            "n_mels": 23,
            # "win_length": 25,
            "hop_length": int(audio_sample_rate * 10/1000.),
            "center": False,
        },
    )
    delta_transform = torchaudio.transforms.ComputeDeltas(win_length=7)

    if mode == 'am' or mode == 'vm':
        feature_means = torch.Tensor(np.load(means_npy))

    if train:
        train_feats = {}
    else:
        features = []
        utt_ids = []
        times = []
    
    print(len(train_list))
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
        noise = torch.randn(audio_tensor.shape) # torch.Size([1, frames])
        snr = torch.Tensor([5])
        audio_tensor = add_noise(audio_tensor, noise, snr) # torch.Size([1, frames])
        mfcc_features = mfcc_transform(audio_tensor) # torch.Size([1, 26, 17613])
        d_mfcc_features = delta_transform(mfcc_features)
        dd_mfcc_features = delta_transform(d_mfcc_features)
        audio_features = torch.cat((mfcc_features, d_mfcc_features, dd_mfcc_features), dim=1)
        audio_features = torch.transpose(torch.squeeze(audio_features, dim=0), 0, 1)
        # audio_features = mfcc_transform(audio_tensor)
        # audio_features = torch.transpose(torch.squeeze(audio_features, dim=0), 0, 1)
        # print(audio_features.shape) # torch.Size([17613, 26])
        
        # check length of original unpadded audio so we don't include the silence at the end
        original_audio, sr = torchaudio.load(os.path.splitext(os.path.join(audio_corpus, video_name))[0] + ".wav")
        num_audio_feature_frames = (original_audio.shape[1] - int(sr * 25/1000.))//int(sr * 10/1000.) + 1
        # print(num_audio_feature_frames)

        if len(video_features) < (num_audio_feature_frames*10+25 + 500) // (1000/video_fps):
            print(f"{os.path.splitext(video_name)[0]} missing video frames")
            print(len(audio_features), num_audio_feature_frames)
            print(len(video_features))
            continue
        
        if not train:
            data = []
        # match video frames to audio frames
        for a_idx in range(num_audio_feature_frames):
            ## NEW FOR BUCKEYE
            video_name_stem = Path(video_name).stem
            video_name_stem = video_name_stem[3:5] + ('1' if video_name_stem[5] == 'a' else '2') + video_name_stem[7:]
            ## 
            frame_data = frame_transform(a_idx, video_features, audio_features, video_name_stem, video_fps)
            if frame_data is None: # video frame is out of range
                continue

            if mode == 'a':
                frame_data[audio_features.shape[1]] = frame_data[-2]
                frame_data[audio_features.shape[1]+1] = frame_data[-1]
                frame_data = frame_data[:audio_features.shape[1]+2]
            elif mode == 'v':
                frame_data = frame_data[audio_features.shape[1]:]
            elif mode == 'an':
                # frame_data = frame_data with visual features masked out
                frame_data[audio_features.shape[1]:-2] = 0 
            elif mode == 'vn':
                # frame_data = frame_data with audio features masked out
                frame_data[:audio_features.shape[1]] = 0 
            elif mode == 'am':
                vdim = len(feature_means[audio_features.shape[1]:])
                frame_data[audio_features.shape[1]:-2] = torch.cat((torch.zeros((vdim,)), feature_means[audio_features.shape[1]:], torch.zeros((vdim,))))
            elif mode == 'vm':
                frame_data[:audio_features.shape[1]] = feature_means[:audio_features.shape[1]]
            # if mode == 'av' change nothing

            if train:
                frame_data = frame_data.detach().cpu().numpy().astype('float64')
                label = f"file{int(frame_data[-2])}_window{int(frame_data[-1])}"
                train_feats[label] = frame_data[:-2]
            else:
                data.append(torch.unsqueeze(frame_data, dim=0)) # torch.Size([1, 178])
        
        if not train:
            data_tensor = torch.cat(data, dim=0)
            data_ndarray = data_tensor.detach().cpu().numpy().astype('float64') ## FLOAT64 IS NEW
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
    # print(len(train_list))
    pca_model = load(args.model_name)
    video_transform = getattr(video_library, args.video_fn)
    frame_transform = getattr(video_library, args.frame_fn)
    # print(video_transform)
    # print(len(train_list))
    process_videos(train_list, pca_model, video_transform, frame_transform, args.video_corpus, args.audio_corpus, args.split, args.output_file, args.mode, args.means_npy)
    # if args.split == 'train':
    #     process_train_videos(train_list, pca_model, args.video_corpus, args.audio_corpus, args.output_file)
    # elif args.split == 'eval' or args.split == 'test':
    #     process_eval_videos(train_list, pca_model, args.video_corpus, args.audio_corpus, args.output_file)
    # train_data = process_videos(train_list, pca_model, args.video_corpus, args.audio_corpus)
    # np2out(train_data, args.output_file, args.split)
