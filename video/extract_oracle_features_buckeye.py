"""
using pretrained visual features PCA model,
extract audio-visual features from .mp4 videos
"""

import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.io import savemat
from joblib import load
import pickle
import json
import torchaudio
from torchvision.io import read_video
import torch
import os
import video_library # from video_library import video_transform_90x190 as video_transform, frame_transform_5500 as frame_transform
import numpy as np
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for audiovisual feature extraction')
    
    parser.add_argument('train_json', help='Path to json list of training videos')
    parser.add_argument('model_name', help='Path to video model pkl file')
    parser.add_argument('output_file', help='Path to output .mat or .npy file to write features to')
    parser.add_argument('--video_fn', default='video_transform_7797', help='video transform function name')
    parser.add_argument('--frame_fn', default='frame_transform_offset_frontpad_120', help='frame transform function name')
    parser.add_argument('--video_corpus', default='/om2/user/szhi/corpora/buckeye_synthetic_video')
    parser.add_argument('--audio_corpus', default='/om2/user/szhi/corpora/buckeye_segments')
    parser.add_argument('--split', default='train') # can also be 'test' or 'eval'
    parser.add_argument('--mode', default='av', help='Modality/ies of features to extract. Can be "a" (audio only), "v", "an" (audio with noise for visual), "vn", or "av".')

    args = parser.parse_args()
    return args


def read_train_list(train_json):
    with open(train_json, 'r') as f:
        train_list = json.load(f)
    print(len(train_list))
    return train_list

def aidx2phone(a_idx, alignment_df):
    a_time = ( 10*a_idx + 25.0/2 ) / 1000.0
    # print(a_idx, a_time)
    # print(alignment_df)
    phone_match = alignment_df[alignment_df['Begin'] <= a_time]
    # print(phone_match)
    phone_match = phone_match[phone_match['End'] >= a_time]
    # print(phone_match)
    if len(phone_match) == 0:
        return 'SIL'
    phone_label = phone_match['Label'].iloc[0]
    post_onset_time = a_time - phone_match['Begin'].iloc[0]
    return phone_label, post_onset_time


STOPS = set(['P','B','K','T','D','G'])
def phone2feat(phone_label, post_onset_time):
    # print(phone_label)
    phones = ['AA','AE','AH','AO','AW','AY','EH','ER','EY','IH','IY','OW','OY','UH','UW',\
              'B','CH','D','DH','F','G','HH','JH','K','L','M','N','NG','P','R','S','SH','T','TH','V','W','Y','Z','ZH',]
    try:
        idx = phones.index(phone_label)
    except:
        idx = len(phones)
    
    # onehot = torch.nn.functional.one_hot(torch.Tensor([idx]).long(), num_classes=len(phones)+1)
    # return onehot[0]

    # # be1375fy_oracle_brokenstops20
    # # onehot = torch.empty((len(phones)+3,)) # normal idxs, other bucket, P/B bucket, K/G/T/D bucket
    # onehot = torch.normal(mean=torch.zeros((len(phones)+3,)), std=torch.zeros((len(phones)+3,))+0.2)
    # if (phone_label == 'P' or phone_label == 'B') and post_onset_time < 20:
    #     onehot[-2] = 1
    # elif phone_label in STOPS and post_onset_time < 20:
    #     onehot[-1] = 1
    # else:
    #     onehot[idx] = 1
    # return onehot

    # be1375fy_oracle_brokenstops30
    # onehot = torch.empty((len(phones)+3,)) # normal idxs, other bucket, P/B bucket, K/G/T/D bucket
    onehot = torch.normal(mean=torch.zeros((len(phones)+3,)) + 4, std=torch.zeros((len(phones)+3,))+2)
    if (phone_label == 'P' or phone_label == 'B') and post_onset_time < 30:
        onehot[-2] = 10
    elif phone_label in STOPS and post_onset_time < 30:
        onehot[-1] = 10
    else:
        onehot[idx] = 10
    return onehot

    # # be1375fy_oracle_onehot
    # onehot = torch.empty((len(phones)+1,))
    # onehot[idx] = 1
    # return onehot

    # scalar = torch.Tensor([idx])
    # return scalar

    # onehot = torch.nn.functional.one_hot(torch.Tensor([idx]).long(), num_classes=len(phones)+1)
    # return torch.concat([onehot[0], onehot[0]], dim=-1)

def frame_transform_offset_frontpad(a_idx, video_features, audio_features, video_name_stem, video_fps, offset=120, frontpad=500):
    def a2v_idx_offset(audio_idx, video_fps, offset):
        """ does not necessarily return an index >= 0 """
        time_ms = audio_idx * 10 + 25/2 - offset
        time_between_vidx = 1000 / video_fps
        video_idx = int(time_ms // time_between_vidx)
        return video_idx

    v_idx = a2v_idx_offset(int(a_idx+frontpad/10), video_fps, offset)
    if v_idx < 0:
        return None

    # append audio slice + video slice to data
    frame_data = torch.cat(
        [audio_features[int(a_idx+frontpad/10)], 
            torch.zeros((42,)),
            torch.Tensor([int(video_name_stem)]), 
            torch.Tensor([a_idx])], 
            dim=-1,
    ) # torch.Size([178])
    return frame_data

def process_videos(
    train_list,
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
    
    for video_name_idx, video_name in enumerate(tqdm(train_list)):
        if not os.path.exists(os.path.splitext(os.path.join(video_corpus, video_name))[0] + ".mp4"):
            print(f"Missing file {os.path.splitext(video_name)[0]}")
            continue

        # read corresponding alignment file
        alignment_corpus = '/om2/user/szhi/corpora/buckeye_segments'
        alignment_file = os.path.join(alignment_corpus, video_name[:-3]+"csv")
        try:
            alignment_df = pd.read_csv(alignment_file)
        except FileNotFoundError:
            print(alignment_file)
            continue
        alignment_df = alignment_df[alignment_df['Type'] == 'phones']
        
        video_info = read_video(os.path.splitext(os.path.join(video_corpus, video_name))[0] + ".mp4", pts_unit='sec')
        video_tensor = video_info[0] # BHWC, torch.Size([742, 1080, 1920, 3])
        video_features = video_transform(video_tensor)
        # video_features = pca_model.transform(video_features) # 60fps
        # video_features = torch.Tensor(video_features)
        # # print(video_features.shape) # Bx50
        video_fps = video_info[2]['video_fps']
        
        audio_tensor = video_info[1] # torch.Size([1, frames])
        audio_features = mfcc_transform(audio_tensor) # torch.Size([1, 26, 17613])
        audio_features = torch.transpose(torch.squeeze(audio_features, dim=0), 0, 1)
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
            phone, post_onset_time = aidx2phone(a_idx, alignment_df)
            video_name_stem = Path(video_name).stem
            video_name_stem = video_name_stem[3:5] + ('1' if video_name_stem[5] == 'a' else '2') + video_name_stem[7:]
            frame_data = frame_transform(a_idx, video_features, audio_features, video_name_stem, video_fps)

            # frame_data = phone2feat(aidx2phone(a_idx, alignment_df))
            # print(frame_data)
            if frame_data is None: # video frame is out of range
                continue

            oracle_video_full = torch.empty((42,))
            oracle_video = phone2feat(phone, post_onset_time)
            oracle_video_full[:len(oracle_video)] = oracle_video
            frame_data[audio_features.shape[1]:-2] = oracle_video_full

            # frame_data = torch.cat(
            #     [frame_data, 
            #     torch.Tensor([int(Path(video_name).stem)]), 
            #     torch.Tensor([a_idx])], 
            #     dim=-1,
            # )

            # if mode == 'a':
            #     # new_frame_data = torch.cat([frame_data[:audio_features.shape[1]],frame_data[-2:]],dim=0)
            #     # del frame_data
            #     # frame_data = new_frame_data
            #     frame_data[audio_features.shape[1]] = frame_data[-2]
            #     frame_data[audio_features.shape[1]+1] = frame_data[-1]
            #     frame_data = frame_data[:audio_features.shape[1]+2]
            # elif mode == 'v':
            #     frame_data = frame_data[audio_features.shape[1]:]
            # elif mode == 'an':
            #     # frame_data = frame_data with visual features masked out
            #     frame_data[audio_features.shape[1]:-2] = 0 # TODO: what is the best mask value
            # elif mode == 'vn':
            #     # frame_data = frame_data with audio features masked out
            #     frame_data[:audio_features.shape[1]] = 0 # TODO: what is the best mask value
            # # if mode == 'av' change nothing

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

    # pca_model = load(args.model_name)
    pca_model = None
    video_transform = getattr(video_library, args.video_fn)
    frame_transform = getattr(video_library, args.frame_fn)

    process_videos(train_list, video_transform, frame_transform_offset_frontpad, args.video_corpus, args.audio_corpus, args.split, args.output_file, args.mode)

