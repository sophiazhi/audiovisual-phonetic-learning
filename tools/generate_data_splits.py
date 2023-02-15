import argparse
import json
import random
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for ABX task creation')
    
    parser.add_argument('id', help='Identifier for this data split')
    parser.add_argument('n_pretrain', type=int, help='Number of pretrain videos')
    parser.add_argument('n_train', type=int, help='Number of train videos')
    parser.add_argument('n_test', type=int, help='Number of test videos')
    
    args = parser.parse_args()
    return args

def get_videos():
    # with 5500 total, do (2500, 2000, 1000)
    previous_sample_lists = ['audio_jsons/first100.json', 'audio_jsons/first900.json', 'audio_jsons/second1000.json', 'audio_jsons/third1k.json', 'audio_jsons/fourth2500.json',]
    
    previous_samples = set()
    for l in previous_sample_lists:
        with open(l, 'r') as f:
            previous_samples.update(json.load(f))
    
    previous_samples = list(previous_samples)
    return previous_samples

def split_videos(previous_samples, args):
    random.shuffle(previous_samples)
    
    pretrain = previous_samples[:args.n_pretrain]
    train = previous_samples[args.n_pretrain:args.n_pretrain+args.n_train]
    test = previous_samples[args.n_pretrain+args.n_train:args.n_pretrain+args.n_train+args.n_test]
    
    pretrain_path = os.path.join(args.id, "pretrain.json")
    train_path = os.path.join(args.id, "train.json")
    test_path = os.path.join(args.id, "test.json")
    
    os.mkdir(args.id)
    
    with open(pretrain_path, 'w') as f:
        json.dump(pretrain, f, indent=2)
    print(f"Saved {pretrain_path}")
    
    with open(train_path, 'w') as f:
        json.dump(train, f, indent=2)
    print(f"Saved {train_path}")
    
    with open(test_path, 'w') as f:
        json.dump(test, f, indent=2)
    print(f"Saved {test_path}")

if __name__ == "__main__":
    args = parse_args()
    previous_samples = get_videos()
    split_videos(previous_samples, args)