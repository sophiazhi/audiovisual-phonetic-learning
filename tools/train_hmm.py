import numpy as np
import pickle
from hmmlearn import hmm
import argparse
# from scipy.io import loadmat
from joblib import dump, load
import h5features as h5f
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for audiovisual feature extraction')
    
    parser.add_argument('features_npy', help='Path to pickle file of train or eval features')
    parser.add_argument('model_name', help='Path to video model pkl file')
    parser.add_argument('step', default='train', help='train or eval')

    parser.add_argument('--posteriors_file', default='hmm_test/posteriors.features')
    parser.add_argument('--phone_states', default=None, help='csv of phones and theoretical number of states')

    args = parser.parse_args()
    return args

def pkl2np(pkl_file):
    with open(pkl_file, 'rb') as f:
        utt_ids, times, features = pickle.load(f)
    return utt_ids,times,features

def train(utt_ids, times, features, model_name, n_components=40, n_mix=3):
    X = np.concatenate(features)
    lengths = np.array([len(x) for x in features])

    model = hmm.GMMHMM(n_components=n_components, n_mix=n_mix)
    model.fit(X, lengths=lengths)
    dump(model, model_name)

def eval(utt_ids, times, features, model_name, output_file):
    X = np.concatenate(features)
    lengths = np.array([len(x) for x in features])

    model = load(model_name)
    print(model.weights_.shape)
    log_prob, posteriors = model.score_samples(X, lengths=lengths)
    print(log_prob)
    print(posteriors.shape)
    
    splits = np.cumsum(lengths)
    posteriors_list = np.split(posteriors, splits)[:-1] # last element is empty
    h5f.write(output_file, 'features', utt_ids, times, posteriors_list)

if __name__ == '__main__':
    args = parse_args()
    u,t,f = pkl2np(args.features_npy)

    if args.step == 'train':
        if args.phone_states is not None:
            phone_states = pd.read_csv(args.phone_states)
            print(phone_states['states_gold'].sum())
            train(u,t,f,args.model_name,n_components=phone_states['states_gold'].sum(), n_mix=1)
        else:
            train(u,t,f,args.model_name)
    elif args.step == 'eval':
        eval(u,t,f,args.model_name, output_file=args.posteriors_file)
