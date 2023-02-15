import h5features as h5f
import argparse
import numpy as np

'''
use venv `h5features`
'''

def convert(mat_file, output_file):
    with h5f.Converter(output_file, groupname='features') as converter:
        converter.convert(mat_file)

def np2h5(npy_file, output_file):
    '''https://github.com/sophiazhi/perceptual-tuning-pnas/blob/main/tools/kaldi_feats/kaldif2h5f.py'''
    data = np.load(npy_file)
    print(np.unique(data[:,-2]))
    features = []
    utt_ids = []
    times = []
    
    curr_utt_start = 0
    prev_utt_id = None
    
    for i in range(len(data)):
        utt_id = data[i, -2]
        
        if prev_utt_id is not None and utt_id != prev_utt_id:
            prev_utt_start = curr_utt_start  # inclusive
            prev_utt_end = i                 # exclusive
            
            features.append(data[prev_utt_start:prev_utt_end, :-2])
            utt_ids.append(int(prev_utt_id))
            times.append(data[prev_utt_start:prev_utt_end, -1] * 0.010 + 0.0125)
            
            curr_utt_start = i
            
        prev_utt_id = utt_id
        
    features.append(data[curr_utt_start:, :-2])
    utt_ids.append(int(prev_utt_id))
    times.append(data[curr_utt_start:, -1] * 0.010 + 0.0125)
    
    # print(utt_ids)
    
    sorted_zip = sorted(zip(utt_ids, times, features), key=lambda triple: triple[0])
    utt_ids = [str(u) for u,_,_ in sorted_zip]
    times = [t for _,t,_ in sorted_zip]
    features = [f for _,_,f in sorted_zip]
    
    # print(utt_ids)
    
    h5f.write(output_file, 'features', utt_ids, times, features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mat_file')
    parser.add_argument('output_file')
    args = parser.parse_args()
    
    # convert(args.mat_file, args.output_file)
    np2h5(args.mat_file, args.output_file)
