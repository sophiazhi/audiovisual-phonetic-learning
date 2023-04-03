import argparse
from pathlib import Path
# import random
import json
import pandas as pd
import os
import ABXpy.task
import ABXpy.distances.distances as dis
import ABXpy.score as sco
import ABXpy.analyze as ana
import ABXpy.distances.metrics.dtw as dtw
import ABXpy.distances.metrics.kullback_leibler as kl
import ABXpy.distances.metrics.cosine as cos
import scipy.spatial.distance as euc
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for ABX task creation')
    
    parser.add_argument('eval_json', help='Path to json list of eval videos')
    parser.add_argument('feat_file', help='Path to eval set features file')
    # parser.add_argument('task_file', default='task.abx', help='File to write task to')
    parser.add_argument('res_id', help='identifier for task results')
    # parser.add_argument('res_folder', default='root/eval/abx', help=('Result folder (must contain'
    #                                         'distances, scores and results'
    #                                         'subfolders)'))
    
    parser.add_argument('--av_offset', default=0, type=int, help='video-audio offset used in feature extraction')
    parser.add_argument('--alignment_corpus', default='/om2/user/szhi/corpora/childes_synthetic_audio_alignment', help='Path to corpus of audio-text alignment .csvs')
    parser.add_argument('--distance', default='kl', help='kl or cos or dur or euc')
    parser.add_argument('--normalized', default=True, help='if true, take mean distance along dtw path length instead of sum')
    parser.add_argument('--n_cpu', default=1)
    
    args = parser.parse_args()
    return args


def get_file_list(eval_json, num_files=5000):
    with open(eval_json, 'r') as f:
        eval_list = json.load(f)
    
    return eval_list


def _remove_stress(phoneme):
    return phoneme[:-1] if phoneme[-1].isdigit() else phoneme


def process_files(file_list, alignment_corpus, av_offset):
    cumulative_df = []

    missing_files = set(pd.read_csv('/om2/user/szhi/perceptual-tuning-pnas/missing.txt', header=None)[0])
    
    for filename in file_list:
        if filename[:-4] in missing_files:
            print("found missing file {}".format(filename))
            continue
        alignment_file = os.path.join(alignment_corpus, os.path.splitext(filename)[0] + ".csv")
        df = pd.read_csv(alignment_file, index_col=False)
        df = df[df['Type'] == 'phones'] # some entries in the csv are word alignments
        video_name_stem = Path(filename).stem
        # video_name_stem = video_name_stem[3:5] + ('1' if video_name_stem[5] == 'a' else '2') + video_name_stem[7:]
        df['#file'] = video_name_stem
        
        df = df.reset_index(drop=True)
        df['prev-phone'] = pd.concat([pd.Series(['<s>']), df['Label']], ignore_index=True).drop(index=len(df)).reset_index(drop=True)
        df['next-phone'] = pd.concat([df['Label'], pd.Series(['<e>'])], ignore_index=True).drop(index=0).reset_index(drop=True)
        
        # each phone range must have at least one audio window CENTERED within the range
        # for offset=0, drop phonemes with onset 0 and offset 10 because it will not have a corresponding mfcc feature
        # df = df.drop(df[(df['Begin'] == 0.0) & (df['End'] == 0.010)].index) # worked for offset=0
        ### audio at offset ms --> a_idx offset/10 --> 0th video
        # audio_idx * 10 + 25/2 - offset >= 0 --> audio_idx >= ceil((offset-12.5)/10)
        # first audio idx at ceil((offset-12.5)/10) --> first audio window centered at 10*ceil((offset-12.5)/10) + 25/2
        df = df.drop(df[df['End']*1000 <= 10*math.ceil((av_offset-12.5)/10) + 25/2].index) # end of phone must be greater than center of the first audio window
        # df = df.drop(df[(df['Begin'] + df['End'])/2 < 14*0.010+0.025/2].index) # worked for offset=150 but probably not the most comprehensive
        # start of phone must be less than center of last audio window
        last_aidx = (df['End'].max() - 0.025)//0.01
        # if video_name_stem == 's0101a_018':
        #     print(video_name_stem)
        #     print(last_aidx)
        #     print(df['End'].max())
        #     print(df)
        df = df.drop(df[df['Begin'] >= 0.010*last_aidx + 0.0125].index) 
        # if video_name_stem == 's0101a_018':
        #     print(df)

        if video_name_stem == 's2602a_046':
            # TODO: remove phones that don't contain the center of an audio window within its range
            continue

        cumulative_df.append(df)
    
    cumulative_df = pd.concat(cumulative_df, axis=0, ignore_index=True)
    
    cumulative_df['#phone'] = cumulative_df['Label'].apply(_remove_stress)
    cumulative_df['prev-phone'] = cumulative_df['prev-phone'].apply(_remove_stress)
    cumulative_df['next-phone'] = cumulative_df['next-phone'].apply(_remove_stress)

    cumulative_df = cumulative_df.rename(columns={'Begin':'onset', 'End':'offset', 'Speaker':'speaker', })
    cumulative_df = cumulative_df[['#file','onset','offset','#phone','prev-phone','next-phone','speaker']]
    
    return cumulative_df


def df2task(cumulative_df, task_file):
    # convert df into .item file
    cumulative_df.to_csv('data.item', sep='\t')

    ### use .item file to generate Task object/file
    t = ABXpy.task.Task('data.item',
                            on=b'phone',
                            across=[],
                            by=['prev-phone', 'next-phone'], # ['speaker', 'prev-phone', 'next-phone'],
                            verbose=1)
    t.generate_triplets(output=task_file)


if __name__ == "__main__":
    args = parse_args()
    
    '''(feat_file, task_file, dis_file, score_file, result_file, distance,
            normalized):
    """
    Results are saved in:
        $res_folder/distances/'$res_id'.distances
        $res_folder/scores/'$res_id'.scores
        $res_folder/results/'$res_id'.txt
    """
    '''
    assert args.distance in ['kl', 'cos', 'dur', 'euc'], \
        "Distance function {} not supported".format(args.distance)
    if args.distance == 'kl':
        frame_dis = kl.kl_divergence
    elif args.distance == 'cos':
        frame_dis = cos.cosine_distance
    elif args.distance == 'euc':
        frame_dis = lambda x, y: euc.cdist(x, y, 'euclidean')
    if args.distance in ['kl', 'cos', 'euc']:
        distance = lambda x, y, normalized: dtw.dtw(x, y, frame_dis, normalized=normalized)
    else:
        distance = lambda x, y, normalized: np.abs(x.shape[0]-y.shape[0])
    
    
    task_file = os.path.join(args.res_id, args.res_id + ".abx")
    dis_file = os.path.join(args.res_id, args.res_id + ".distance")
    score_file = os.path.join(args.res_id, args.res_id + ".scores")
    result_file = os.path.join(args.res_id, args.res_id + ".txt")
    
    file_list = get_file_list(args.eval_json)
    df = process_files(file_list, args.alignment_corpus, args.av_offset)
    df2task(df, task_file)
    
    dis.compute_distances(args.feat_file, '/features/', task_file, dis_file,
                              distance, normalized=args.normalized, n_cpu=args.n_cpu)
    sco.score(task_file, dis_file, score_file)
    ana.analyze(task_file, score_file, result_file)
