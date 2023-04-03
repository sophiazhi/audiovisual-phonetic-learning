#!/bin/bash                      
#SBATCH -t 48:00:00
#SBATCH -n 1

id="split5500_av76feats"
corpora_audio_alignment="/om2/user/szhi/corpora/childes_synthetic_audio_alignment"
data_test="split5500/test.json"

cd /om2/user/szhi/perceptual-tuning-pnas

### switch to abx_eval conda env ###
source /om2/user/szhi/miniconda3/etc/profile.d/conda.sh
conda activate abx_eval

# convert eval .npy to .features with mat2h5
python tools/mat2h5.py $id/${id}_eval.npy $id/${id}_eval.features
# extract_posteriors 
python tools/dpgmm/extract_posteriors.py $id/${id}_eval.features $id/${id}_1501-final.mat $id/${id}_eval_posteriors.features
# abxpy_eval
python eval/abxpy_eval.py $data_test $id/${id}_eval_posteriors.features $id --alignment_corpus $corpora_audio_alignment

conda deactivate
### end abx_eval conda env ###