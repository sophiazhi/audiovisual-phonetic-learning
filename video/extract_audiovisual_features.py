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
    parser.add_argument('--video_fn', default='video_transform_90x190', help='video transform function name')
    parser.add_argument('--frame_fn', default='frame_transform_5500', help='frame transform function name')
    parser.add_argument('--video_corpus', default='../corpora/childes_synthetic_video_example60fps/')
    parser.add_argument('--audio_corpus', default='../corpora/childes_synthetic_audio/')
    parser.add_argument('--split', default='train') # can also be 'test' or 'eval'
    parser.add_argument('--mode', default='av', help='Modality/ies of features to extract. Can be "a" (audio only), "v", "an" (audio with noise for visual), "vn", or "av".')

    args = parser.parse_args()
    return args


def read_train_list(train_json):
    with open(train_json, 'r') as f:
        train_list = json.load(f)
    print(len(train_list))
    return train_list


'''def a2v_idx(audio_idx, video_fps):
    time_ms = audio_idx * 10 + 25/2 # offset some number between 0 and 25
    time_between_vidx = 1000 / video_fps
    video_idx = int(time_ms // time_between_vidx)
    return video_idx
'''

'''def process_videos_old(train_list, pca_model, video_corpus, audio_corpus):
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
'''

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
    if split == 'train':
        train = True
    elif split == 'test' or split == 'eval':
        train = False
    
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

    if train:
        train_feats = {}
    else:
        features = []
        utt_ids = []
        times = []
    
    print(len(train_list))
    for video_name_idx, video_name in enumerate(tqdm(train_list)):
        video_info = read_video(os.path.splitext(os.path.join(video_corpus, video_name))[0] + ".mp4", pts_unit='sec')
        video_tensor = video_info[0] # BHWC, torch.Size([742, 1080, 1920, 3])
        video_features = video_transform(video_tensor)
        video_features = pca_model.transform(video_features) # 60fps
        video_features = torch.Tensor(video_features)
        # print(video_features.shape) # Bx50
        video_fps = video_info[2]['video_fps']
        
        audio_tensor = video_info[1] # torch.Size([1, frames])
        audio_features = mfcc_transform(audio_tensor) # torch.Size([1, 26, 17613])
        audio_features = torch.transpose(torch.squeeze(audio_features, dim=0), 0, 1)
        # print(audio_features.shape) # torch.Size([17613, 26])
        
        # check length of original unpadded audio so we don't include the silence at the end
        original_audio, sr = torchaudio.load(os.path.splitext(os.path.join(audio_corpus, video_name))[0] + ".wav")
        num_audio_feature_frames = (original_audio.shape[1] - int(sr * 25/1000.))//int(sr * 10/1000.) + 1
        # print(num_audio_feature_frames)
        
        if not train:
            data = []
        # match video frames to audio frames
        for a_idx in range(num_audio_feature_frames):
            frame_data = frame_transform(a_idx, video_features, audio_features, Path(video_name).stem, video_fps)
            if frame_data is None: # video frame is out of range
                continue

            if mode == 'a':
                # new_frame_data = torch.cat([frame_data[:audio_features.shape[1]],frame_data[-2:]],dim=0)
                # del frame_data
                # frame_data = new_frame_data
                frame_data[audio_features.shape[1]] = frame_data[-2]
                frame_data[audio_features.shape[1]+1] = frame_data[-1]
                frame_data = frame_data[:audio_features.shape[1]+2]
            elif mode == 'v':
                frame_data = frame_data[audio_features.shape[1]:]
            elif mode == 'an':
                # frame_data = frame_data with visual features masked out
                frame_data[audio_features.shape[1]:-2] = 0 # TODO: what is the best mask value
            elif mode == 'vn':
                # frame_data = frame_data with audio features masked out
                frame_data[:audio_features.shape[1]] = 0 # TODO: what is the best mask value
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

'''def process_train_videos(train_list, pca_model, video_corpus, audio_corpus, output_file):
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
    
    train_list = {} ## NEW

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
            # data.append(torch.unsqueeze(frame_data, dim=0)) # torch.Size([1, 178]) ## OLD
            label = f"file{int(frame_data[-2])}_window{int(frame_data[-1])}"
            train_list[label] = frame_data[:-2]
    
    ## OLD
    # data_tensor = torch.cat(data, dim=0)
    # data_ndarray = data_tensor.detach().cpu().numpy()
    # print(data_ndarray.shape)
    # return data_ndarray

    ## NEW
    savemat(output_file, train_list)
'''

'''def process_eval_videos(train_list, pca_model, video_corpus, audio_corpus, output_file):
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
    
    features = []
    utt_ids = []
    times = []

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
        
        data = []
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
        
        ## NEW
        data_tensor = torch.cat(data, dim=0)
        data_ndarray = data_tensor.detach().cpu().numpy()
        features.append(data_ndarray[:, :-2])
        utt_ids.append(Path(video_name).stem)
        times.append(data_ndarray[:, -1] * 0.010 + 0.0125)
    
    sorted_zip = sorted(zip(utt_ids, times, features), key=lambda triple: triple[0])
    utt_ids = [str(u) for u,_,_ in sorted_zip]
    times = [t for _,t,_ in sorted_zip]
    features = [f for _,_,f in sorted_zip]

    # dump((utt_ids, times, features), output_file, protocol=2) 
    with open(output_file, 'wb') as f:
        pickle.dump((utt_ids, times, features), f, protocol=2)
'''

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
    process_videos(train_list, pca_model, video_transform, frame_transform, args.video_corpus, args.audio_corpus, args.split, args.output_file, args.mode)
    # if args.split == 'train':
    #     process_train_videos(train_list, pca_model, args.video_corpus, args.audio_corpus, args.output_file)
    # elif args.split == 'eval' or args.split == 'test':
    #     process_eval_videos(train_list, pca_model, args.video_corpus, args.audio_corpus, args.output_file)
    # train_data = process_videos(train_list, pca_model, args.video_corpus, args.audio_corpus)
    # np2out(train_data, args.output_file, args.split)
