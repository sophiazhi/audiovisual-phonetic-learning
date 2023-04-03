import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import os

VOWELS = ['AA','AE','AH','AO','AW','AY','EH','ER','EY','IH','IY','OW','OY','UH','UW','VN']
CONSONANTS = ['B','CH','D','DH','F','G','HH','JH','K','L','M','N','NG','P','R','S','SH','T','TH','V','W','Y','Z','ZH','NX','TQ','EM','EN','DX','EL']
OTHER = ['spn']
SUBCLASS_MAP = {'AA': 'vowel','AE': 'vowel','AH': 'vowel','AO': 'vowel','EH': 'vowel','ER': 'vowel','EY': 'vowel','IH': 'vowel','IY': 'vowel','OY': 'vowel','UH': 'vowel','UW': 'vowel', 'VN': 'vowel',
               'AY': 'diphthong','AW': 'diphthong','OW': 'diphthong',
                'B': 'stop','D': 'stop','G': 'stop','P': 'stop','T': 'stop','K': 'stop', 'TQ': 'stop',
                'M': 'nasal','N': 'nasal','NG': 'nasal','NX': 'nasal','EM': 'nasal','EN': 'nasal',
                'V': 'fricative','DH': 'fricative','Z': 'fricative','ZH': 'fricative','F': 'fricative','TH': 'fricative','S': 'fricative','SH': 'fricative','HH': 'fricative',
                'L': 'approximant','EL': 'approximant','R': 'approximant','W': 'approximant','WH': 'approximant','Y': 'approximant',
                'CH': 'affricate','JH': 'affricate',
                'DX': 'flap',
               }

# dx, nx, tq, er, em, vn, EN added for buckeye

def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for ABX score analysis')
    
    parser.add_argument('results_txt', help='Path to input txt file of ABX scores')
    parser.add_argument('analysis_txt', help='Path to output txt file of aggregated scores')

    args = parser.parse_args()
    return args


def read_results(results_txt):
    results_df = pd.read_csv(results_txt, sep='\t')
    results_df['phone_1'] = results_df['phone_1'].str.upper()
    results_df['phone_2'] = results_df['phone_2'].str.upper()
    results_df['by'] = results_df['by'].str.upper()
    results_df = results_df[results_df['phone_1'].isin(VOWELS+CONSONANTS)]
    results_df = results_df[results_df['phone_2'].isin(VOWELS+CONSONANTS)]
    return results_df


def plot_histogram(results_df, png_name, title='ABX score distribution', threshold_counts=None):
    plt.hist(results_df['score'], density=False, bins=30)
    if threshold_counts:
        above, equal, below = threshold_counts
        annotation = "{} pairs below 0.5 \n{} pairs at 0.5 \n{} pairs above 0.5 ".format(below, equal, above)
        plt.text(0.2,0.7,annotation, transform=plt.gcf().transFigure)
    plt.ylabel('Number of AB pairs in context')
    plt.xlabel('ABX score')
    plt.xticks([0.1*x for x in range(11)])
    plt.title(title)
    plt.savefig('{}.png'.format(png_name))
    plt.clf()


def count_threshold(results_df, threshold=0.5):
    better = len(results_df[results_df['score'] > threshold])
    same = len(results_df[results_df['score'] == threshold])
    worse = len(results_df[results_df['score'] < threshold])
    
    return (better, same, worse)


def weighted_score(results_df):
    return np.average(results_df['score'], weights=results_df['n'])


def confusion_matrix_simple(results_df, png_name):
    def label(phone):
        if phone in VOWELS:
            return 'vowel'
        if phone in CONSONANTS:
            return 'consonant'
        return 'other'
    _confusion_matrix(results_df, label, png_name)

def confusion_matrix_detailed(results_df, png_name):
    def label(phone):
        return SUBCLASS_MAP[phone]
    _confusion_matrix(results_df, label, png_name)

def confusion_matrix_phones(results_df, png_name):
    def label(phone):
        return phone
    _confusion_matrix(results_df, label, png_name, figsize=(20,20))

def _confusion_matrix(results_df, phone2label, png_name, figsize=(8,6), save_counts=True):
    # df_flipped = results_df.rename(columns={'phone_1': 'phone_2', 'phone_2': 'phone_1'})
    # df_labeled = pd.concat((results_df, df_flipped))
    df_labeled = results_df.copy()

    df_labeled['label_1'] = df_labeled['phone_1'].apply(phone2label)
    df_labeled['label_2'] = df_labeled['phone_2'].apply(phone2label)
    df_group = df_labeled.groupby(['label_1', 'label_2'])
        
    A_labels = sorted(df_labeled['label_1'].unique())
    B_labels = sorted(df_labeled['label_2'].unique())

    # scores = df_group['score'].mean().values.reshape((len(A_labels), len(B_labels)))
    scores = np.zeros((len(A_labels), len(B_labels)))
    scores[:] = np.nan
    counts = [['' for bi in range(len(B_labels))] for ai in range(len(A_labels))]
    for group in df_group:
        A, B = group[0]
        aidx, bidx = A_labels.index(A), B_labels.index(B)
        scores[aidx,bidx] = group[1]['score'].mean()
        counts[aidx][bidx] = f"{group[1]['score'].count()}\n({len(group[1]['by'].unique())})"


    fig, ax = plt.subplots(figsize=figsize)
    plt.set_cmap('bwr_r')
    im = ax.imshow(scores.T, vmin=0, vmax=1)

    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.xaxis.set_label_position('top') 
    ax.yaxis.set_label_position('left')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(A_labels)), labels=A_labels)
    ax.set_yticks(np.arange(len(B_labels)), labels=B_labels)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(A_labels)):
        for j in range(len(B_labels)):
            if np.isnan(scores[i,j]):
                continue
            text = ax.text(i,j, round(scores[i, j],3),
                        ha="center", va="center", color="w",
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    ax.set_title("Differentiation scores by AB pair")
    fig.tight_layout()
    plt.savefig('{}.png'.format(png_name))
    plt.clf()

    if save_counts:
        fig, ax = plt.subplots(figsize=figsize)
        plt.set_cmap('bwr_r')
        im = ax.imshow(scores.T, vmin=0, vmax=1)

        ax.set_xlabel("A")
        ax.set_ylabel("B")
        ax.xaxis.set_label_position('top') 
        ax.yaxis.set_label_position('left')

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(A_labels)), labels=A_labels)
        ax.set_yticks(np.arange(len(B_labels)), labels=B_labels)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(A_labels)):
            for j in range(len(B_labels)):
                if np.isnan(scores[i,j]):
                    continue
                text = ax.text(i,j, counts[i][j],
                            ha="center", va="center", color="w",
                            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

        ax.set_title("Number of examples (and unique contexts) by AB pair")
        fig.tight_layout()
        plt.savefig('{}_counts.png'.format(png_name))
        plt.clf()


if __name__ == '__main__':
    args = parse_args()

    path_prefix = os.path.splitext(args.results_txt)[0]

    results_df = read_results(args.results_txt)

    weighted_average = weighted_score(results_df)
    print(weighted_average)

    threshold = 0.5
    counts = count_threshold(results_df, threshold=threshold)
    print(counts)

    plot_histogram(results_df, path_prefix+"_histogram", threshold_counts=counts)
    confusion_matrix_simple(results_df, path_prefix+"_confusion_simple")
    confusion_matrix_detailed(results_df, path_prefix+"_confusion_detailed")
    confusion_matrix_phones(results_df, path_prefix+"_confusion_phones")

    filtered_results_df = results_df[results_df['n'] >= 10]
    filtered_counts = count_threshold(filtered_results_df, threshold=threshold)
    plot_histogram(filtered_results_df, path_prefix+"_histogram_10", title="ABX score distribution for pairs with 10+ examples", threshold_counts=filtered_counts)


    with open(args.analysis_txt, mode='a+') as f:
        f.write("Threshold {}: {} better, {} same, {} worse.\n".format(threshold, counts[0], counts[1], counts[2]))
        f.write("Overall weighted average score: {}.\n".format(weighted_average))