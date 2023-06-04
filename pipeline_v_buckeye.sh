#!/bin/bash                      
#SBATCH -t 12:00:00
#SBATCH -n 4
#SBATCH --mem 60G
#SBATCH --exclude=node[100-116],node086
#SBATCH --array 9-16

id="beminifyf_vdim"$SLURM_ARRAY_TASK_ID"_4"
corpora_audio="/om2/user/szhi/corpora/buckeye_segments"
corpora_audio_alignment="/om2/user/szhi/corpora/buckeye_segments"
corpora_video="/om2/user/szhi/corpora/buckeye_synthetic_video_new16"
data_pretrain="beminifyf/pretrain.json"
data_train="beminifyf/train.json"
data_test="beminifyf/test.json"
video_fn="video_transform_7797"
offset=120
frame_fn="frame_transform_offset_frontpad_"$offset
video_n_components=$SLURM_ARRAY_TASK_ID
dpgmm_alpha=1

echo $id
mkdir $id

### start wav2lip-local singularity image ###
source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles
module load openmind/singularity/3.4.1
cd /om2/user/szhi/perceptual-tuning-pnas

# # pretrain script
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/pretrain_visual_features_buckeye.py $data_pretrain $id/$id.pkl --video_fn $video_fn --n_components $video_n_components --video_corpus $corpora_video
# # extract audiovisual features for train and eval
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features_buckeye.py $data_train $id/$id.pkl $id/${id}_v_train.mat --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --mode v
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features_buckeye.py $data_test $id/$id.pkl $id/${id}_v_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode v

### end wav2lip-local singularity image ###


# train dpgmm
cp $id/${id}_v_train.mat /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/features/${id}_v_train.mat
mkdir /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/${id}_v

module load mit/matlab/2020a
cd /om2/user/szhi/perceptual-tuning-pnas/tools/dpgmm/dpmm_subclusters_2014-08-06/Gaussian/
matlab -nojvm -r "feat_mat = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/features/${id}_v_train.mat'; model_dir = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/${id}_v'; my_run_dpgmm_subclusters(feat_mat, model_dir, 15, true, $dpgmm_alpha); exit;"

cd /om2/user/szhi/perceptual-tuning-pnas
cp /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/${id}_v/1501-final.mat $id/${id}_v_1501-final.mat

### switch to abx_eval conda env ###
source /om2/user/szhi/miniconda3/etc/profile.d/conda.sh
conda activate abx_eval

# convert eval .npy to .features with mat2h5
python tools/mat2h5.py $id/${id}_v_eval.npy $id/${id}_v_eval.features
# extract_posteriors 
python tools/dpgmm/extract_posteriors.py $id/${id}_v_eval.features $id/${id}_v_1501-final.mat $id/${id}_vv_eval_posteriors.features
# abxpy_eval
mkdir ${id}_vv
python eval/abxpy_eval_buckeye.py $data_test $id/${id}_vv_eval_posteriors.features ${id}_vv --av_offset $offset --alignment_corpus $corpora_audio_alignment

conda deactivate

### end abx_eval conda env ###

conda activate h5f_plt 
python eval/analyze_scores.py ${id}_vv/${id}_vv.txt ${id}_vv/${id}_vv_analysis.txt
python eval/analyze_clusters.py $id/${id}_vv_eval_posteriors.features $data_test ${id}_vv --alignment_corpus $corpora_audio_alignment
conda deactivate