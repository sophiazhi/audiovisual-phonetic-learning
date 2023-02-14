"""
take input list (json) of training videos

initialize empty list for all video data
read all videos:
    crop video to mouth region
    greyscale
    reshape to 1D tensor
    append to list
convert list to tensor

fit PCA on training data
save pkl'ed PCA model
optionally, do some eval

"""
import numpy as np
import argparse
from sklearn.decomposition import PCA
from joblib import dump # , load
import json
from torchvision.io import read_video
import torch
import os
from tqdm import tqdm
from video_library import video_transform_5500_230212 as video_transform

def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for video feature extraction')
    
    parser.add_argument('train_json', help='Path to json list of training videos') # input file name
    parser.add_argument('model_name', help='Path to output model pkl file') # model file name
    parser.add_argument('--n_components', type=int, default=50, help='Number of PCA components')
    parser.add_argument('--video_corpus', default='corpora/childes_synthetic_video_60fps/')

    args = parser.parse_args()
    return args

def read_train_list(train_json):
    with open(train_json, 'r') as f:
        train_list = json.load(f)
    return train_list

def process_videos(train_list, video_corpus):
    video_data = []
    for video_name in tqdm(train_list):
        video_info = read_video(os.path.splitext(os.path.join(video_corpus, video_name))[0] + ".mp4", pts_unit='sec')
        video_tensor = video_info[0] # BHWC, torch.Size([742, 1080, 1920, 3])
        video_tensor = video_transform(video_tensor)
        video_data.append(video_tensor)
    
    all_video_tensor = torch.cat(video_data, dim=0)
    all_video_ndarray = all_video_tensor.detach().cpu().numpy()
    return all_video_ndarray

def train(train_data, model_name, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
    dump(pca, model_name)

if __name__ == '__main__':
    args = parse_args()
    
    train_list = read_train_list(args.train_json)
    train_data = process_videos(train_list, args.video_corpus)
    train(train_data, args.model_name, args.n_components)