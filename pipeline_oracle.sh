#!/bin/bash                      
#SBATCH -t 24:00:00
#SBATCH -n 10
#SBATCH --mem 20G
#SBATCH -N 1

# use 35G if doing AV feature extraction

id="split5500_oracle_onehot_multi"
corpora_audio="/om2/user/szhi/corpora/childes_synthetic_audio"
corpora_audio_alignment="/om2/user/szhi/corpora/childes_synthetic_audio_alignment"
corpora_video="/om2/user/szhi/corpora/childes_synthetic_video_example60fps"
data_pretrain="split5500/pretrain.json"
data_train="split5500/train.json"
data_test="split5500/test.json"
video_fn="video_transform_90x190"
offset=120
frame_fn="frame_transform_offset_"$offset
video_n_components=20
dpgmm_alpha=1.0

echo $id
mkdir $id

### start wav2lip-local singularity image ###
module load openmind/singularity/3.4.1
cd /om2/user/szhi/perceptual-tuning-pnas

# # pretrain script
# singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/pretrain_visual_features.py $data_pretrain $id/$id.pkl --video_fn $video_fn --n_components $video_n_components --video_corpus $corpora_video
# extract audiovisual features for train and eval
# singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features.py $data_train $id/$id.pkl $id/${id}_train.mat --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video
# singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features.py $data_test $id/$id.pkl $id/${id}_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode av

### end wav2lip-local singularity image ###


# train dpgmm
cp $id/${id}_train.mat /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/features/${id}_train.mat
mkdir /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/$id

module load mit/matlab/2020a
cd /om2/user/szhi/perceptual-tuning-pnas/tools/dpgmm/dpmm_subclusters_2014-08-06/Gaussian/
matlab -nojvm -r "feat_mat = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/features/${id}_train.mat'; model_dir = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/$id'; my_run_dpgmm_subclusters(feat_mat, model_dir, 15, true, $dpgmm_alpha); exit;"

cd /om2/user/szhi/perceptual-tuning-pnas
cp /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/$id/1501-final.mat $id/${id}_1501-final.mat


### switch to abx_eval conda env ###
source /om2/user/szhi/miniconda3/etc/profile.d/conda.sh
conda activate abx_eval

# convert eval .npy to .features with mat2h5
python tools/mat2h5.py $id/${id}_eval.npy $id/${id}_eval.features
# extract_posteriors 
python tools/dpgmm/extract_posteriors.py $id/${id}_eval.features $id/${id}_1501-final.mat $id/${id}_eval_posteriors.features
# abxpy_eval
python eval/abxpy_eval.py $data_test $id/${id}_eval_posteriors.features $id --av_offset $offset --alignment_corpus $corpora_audio_alignment

conda deactivate

## ADD EVAL FOR AV-A+, AV-V+
### end abx_eval conda env ###