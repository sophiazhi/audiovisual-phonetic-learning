import h5features as h5f
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import argparse
import sklearn.metrics
from scipy.stats import entropy

def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for cluster analysis')
    
    parser.add_argument('posteriors_features', help='Path to .features file of posteriors')
    parser.add_argument('eval_json', help='Path to json file of eval utterances')
    parser.add_argument('id')
    parser.add_argument('--alignment_corpus', default='/om2/user/szhi/corpora/buckeye_segments')
    parser.add_argument('--other_posteriors', default=None, help='path to another posteriors .features file for rand score comparison')

    args = parser.parse_args()
    return args

def read_posteriors(posteriors_file):
    with h5f.Reader(posteriors_file, groupname="features") as reader:
        eval_posteriors = reader.read()
    return eval_posteriors

def read_eval_json(eval_json):
    with open(eval_json, 'r') as f:
        eval_list = json.load(f)
    return eval_list

def _atime2phone(a_time, alignment_df):
    # a_time = ( 10*a_idx + 25.0/2 ) / 1000.0
    phone_match = alignment_df[alignment_df['Begin'] <= a_time]
    phone_match = phone_match[phone_match['End'] >= a_time]
    if len(phone_match) == 0:
        return 'SIL'
    phone_label = phone_match['Label'].iloc[0]
    return phone_label

def confusion_matrix(eval_list, eval_posteriors, alignment_corpus, res_id):
    # dictionary mapping phone key to np array of cluster occurrence counts
    counts_by_phone = {}
    for filename in eval_list:
        utt_name = Path(filename).stem
        if utt_name not in eval_posteriors.dict_features():
            continue
            
        alignment_file = os.path.join(alignment_corpus, os.path.splitext(filename)[0] + ".csv")
        df = pd.read_csv(alignment_file, index_col=False)
        df = df[df['Type'] == 'phones'] # some entries in the csv are word alignments

        df = df.reset_index(drop=True)
        
        times = eval_posteriors.dict_labels()[utt_name]
        clusters = eval_posteriors.dict_features()[utt_name].argmax(axis=1)
        
        for i in range(len(times)):
            time, cluster = times[i], clusters[i]
            phone = _atime2phone(time, df)
            if phone not in counts_by_phone:
                counts_by_phone[phone] = np.zeros((eval_posteriors.dict_features()[utt_name].shape[1],))
            counts_by_phone[phone][cluster] += 1
    
    phone_labels = sorted(list(counts_by_phone.keys()))
    cluster_counts = [counts_by_phone[pl] for pl in phone_labels] # phone x cluster
    cluster_array = np.array(cluster_counts).T # cluster x phone
    cluster_array /= cluster_array.sum(axis=1,keepdims=True)

    fig, ax = plt.subplots(figsize=(50,50))
    im = ax.imshow(cluster_array)

    num_clusters = eval_posteriors.dict_features()[Path(eval_list[0]).stem].shape[1]
    ax.set_xticks(np.arange(len(phone_labels)), labels=phone_labels)
    ax.set_yticks(np.arange(num_clusters), labels=[str(int(i)) for i in range(num_clusters)])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(phone_labels)):
        for j in range(num_clusters):
            # if np.isnan(scores[i,j]):
            #     continue
            text = ax.text(i, j, round(cluster_array[j, i],2),
                        ha="center", va="center", color="w",
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="grey")])

    ax.set_title("Cluster assignments")
    fig.tight_layout()
    plt.savefig(f'{res_id}/{res_id}_cluster_assignments.png')
    plt.clf()


    assignments = cluster_array.argmax(axis=1)
    assignments = list(assignments)
    cluster_counts_dict = {phone_labels[i]:assignments.count(i) for i in set(assignments)}
    entropies = entropy(cluster_array, axis=1)
    unweighted_avg_entropy = np.average(entropies)
    weighted_avg_entropy = np.average(entropies, weights=np.array(cluster_counts).T.sum(axis=1))
    with open(f"{res_id}/{res_id}_cluster_eval.json", 'w') as f:
        json.dump({
            'cluster_counts': cluster_counts_dict, 
            'unweighted_average_entropy': unweighted_avg_entropy, 
            'weighted_average_entropy': weighted_avg_entropy
            }, f)
    
def rand_score(eval_list, A_posteriors, B_posteriors):
    # dictionary mapping phone key to np array of cluster occurrence counts
    A_assignments = []
    B_assignments = []
    for filename in eval_list:
        utt_name = Path(filename).stem
        if utt_name not in A_posteriors.dict_features():
            continue
        
        A_clusters = A_posteriors.dict_features()[utt_name].argmax(axis=1)
        B_clusters = B_posteriors.dict_features()[utt_name].argmax(axis=1)
        A_assignments.append(A_clusters)
        B_assignments.append(B_clusters)
    
    all_A = np.concatenate(A_assignments)
    all_B = np.concatenate(B_assignments)
    adjusted_rand = sklearn.metrics.adjusted_rand_score(all_A, all_B)
    return adjusted_rand
        

if __name__ == '__main__':
    args = parse_args()
    eval_posteriors = read_posteriors(args.posteriors_features)
    eval_list = read_eval_json(args.eval_json)

    confusion_matrix(eval_list, eval_posteriors, args.alignment_corpus, args.id)

    if args.other_posteriors is not None:
        other_posteriors = read_posteriors(args.other_posteriors)
        print(rand_score(eval_list, eval_posteriors, other_posteriors))