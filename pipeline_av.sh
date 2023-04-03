#!/bin/bash                      
#SBATCH -t 36:00:00
#SBATCH -n 10
#SBATCH --mem 80G

# use 35G if doing AV feature extraction for childes

id="be1718fy_dpgmm_120"
corpora_audio="/om2/user/szhi/corpora/childes_synthetic_audio"
corpora_audio_alignment="/om2/user/szhi/corpora/childes_synthetic_audio_alignment"
corpora_video="/om2/user/szhi/corpora/childes_synthetic_video_example60fps"
data_pretrain="be1718fy/pretrain.json"
data_train="be1718fy/train.json"
data_test="be1718fy/test.json"
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

# pretrain script
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/pretrain_visual_features.py $data_pretrain $id/$id.pkl --video_fn $video_fn --n_components $video_n_components --video_corpus $corpora_video
# extract audiovisual features for train and eval
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features.py $data_train $id/$id.pkl $id/${id}_train.mat --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features.py $data_test $id/$id.pkl $id/${id}_av_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode av

# eval feature extraction in modes 'an' and 'vn'
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features.py $data_test $id/$id.pkl $id/${id}_an_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode an
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features.py $data_test $id/$id.pkl $id/${id}_vn_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode vn

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
python tools/mat2h5.py $id/${id}_av_eval.npy $id/${id}_av_eval.features
# extract_posteriors 
python tools/dpgmm/extract_posteriors.py $id/${id}_av_eval.features $id/${id}_1501-final.mat $id/${id}_avav_eval_posteriors.features
# abxpy_eval
mkdir ${id}_avav
python eval/abxpy_eval.py $data_test $id/${id}_avav_eval_posteriors.features ${id}_avav --av_offset $offset --alignment_corpus $corpora_audio_alignment

# convert eval .npy to .features with mat2h5
python tools/mat2h5.py $id/${id}_an_eval.npy $id/${id}_an_eval.features
# extract_posteriors 
python tools/dpgmm/extract_posteriors.py $id/${id}_an_eval.features $id/${id}_1501-final.mat $id/${id}_ava_eval_posteriors.features
# abxpy_eval
mkdir ${id}_ava
python eval/abxpy_eval.py $data_test $id/${id}_ava_eval_posteriors.features ${id}_ava --av_offset $offset --alignment_corpus $corpora_audio_alignment

# convert eval .npy to .features with mat2h5
python tools/mat2h5.py $id/${id}_vn_eval.npy $id/${id}_vn_eval.features
# extract_posteriors 
python tools/dpgmm/extract_posteriors.py $id/${id}_vn_eval.features $id/${id}_1501-final.mat $id/${id}_avv_eval_posteriors.features
# abxpy_eval
mkdir ${id}_avv
python eval/abxpy_eval.py $data_test $id/${id}_avv_eval_posteriors.features ${id}_avv --av_offset $offset --alignment_corpus $corpora_audio_alignment

conda deactivate

### end abx_eval conda env ###

conda activate plt 
python eval/analyze_scores.py ${id}_avav/${id}_avav.txt ${id}_avav/${id}_avav_analysis.txt
python eval/analyze_scores.py ${id}_ava/${id}_ava.txt ${id}_ava/${id}_ava_analysis.txt
python eval/analyze_scores.py ${id}_avv/${id}_avv.txt ${id}_avv/${id}_avv_analysis.txt
conda deactivate