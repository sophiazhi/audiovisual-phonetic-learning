#!/bin/bash                      
#SBATCH -t 5:00:00
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH --exclude=node[100-116],node086,dgx001
#SBATCH --array 0-9

# use 80G for AV feature extraction for buckeye

# id="be4620fyf_043023_001_"$SLURM_ARRAY_TASK_ID
# corpora_audio="/om2/user/szhi/corpora/buckeye_segments_short"
# corpora_audio_alignment="/om2/user/szhi/corpora/buckeye_segments_short"
# corpora_video="/om2/user/szhi/corpora/buckeye_synthetic_video_short_new16"
# data_pretrain="be4620fyf/pretrain.json"
# data_train="be4620fyf/train.json"
# data_test="be4620fyf/test.json"
# video_fn="video_transform_7797"
# offset=130
# frame_fn="frame_transform_offset_frontpad_"$offset
# video_n_components=4
# dpgmm_alpha=0.01
id="be1375fy_043023_"$SLURM_ARRAY_TASK_ID
corpora_audio="/om2/user/szhi/corpora/buckeye_segments"
corpora_audio_alignment="/om2/user/szhi/corpora/buckeye_segments"
corpora_video="/om2/user/szhi/corpora/buckeye_synthetic_video_new16"
data_pretrain="be1375fy/pretrain.json"
data_train="be1375fy/train.json"
data_test="be1375fy/test.json"
video_fn="video_transform_7797"
offset=130
frame_fn="frame_transform_offset_frontpad_"$offset
video_n_components=4
dpgmm_alpha=0.01

echo $id
mkdir $id

### start wav2lip-local singularity image ###
source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles
module load openmind/singularity/3.4.1
cd /om2/user/szhi/perceptual-tuning-pnas

# extract audiovisual features for eval
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_noisy_audiovisual_features_buckeye.py $data_test $id/$id.pkl $id/${id}_nav_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode av
## eval feature extraction in modes 'an'
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_noisy_audiovisual_features_buckeye.py $data_test $id/$id.pkl $id/${id}_na_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode a

### end wav2lip-local singularity image ###


### switch to abx_eval conda env ###
source /om2/user/szhi/miniconda3/etc/profile.d/conda.sh
conda activate abx_eval

python tools/mat2h5.py $id/${id}_nav_eval.npy $id/${id}_nav_eval.features
python tools/dpgmm/extract_posteriors.py $id/${id}_nav_eval.features $id/${id}_1501-final.mat $id/${id}_avnav_eval_posteriors.features
mkdir ${id}_avnav
python eval/abxpy_eval_buckeye.py $data_test $id/${id}_avnav_eval_posteriors.features ${id}_avnav --av_offset $offset --alignment_corpus $corpora_audio_alignment

python tools/mat2h5.py $id/${id}_na_eval.npy $id/${id}_na_eval.features
python tools/dpgmm/extract_posteriors.py $id/${id}_na_eval.features $id/${id}_a_1501-final.mat $id/${id}_ana_eval_posteriors.features
mkdir ${id}_ana
python eval/abxpy_eval_buckeye.py $data_test $id/${id}_ana_eval_posteriors.features ${id}_ana --av_offset $offset --alignment_corpus $corpora_audio_alignment

# python tools/dpgmm/extract_posteriors_flex.py $id/${id}_na_eval.features $id/${id}_1501-final.mat $id/${id}_avna_eval_posteriors.features --flex_align left
# mkdir ${id}_avna
# python eval/abxpy_eval_buckeye.py $data_test $id/${id}_avna_eval_posteriors.features ${id}_avna --av_offset $offset --alignment_corpus $corpora_audio_alignment

conda deactivate

### end abx_eval conda env ###

conda activate h5f_plt 
python eval/analyze_scores.py ${id}_avnav/${id}_avnav.txt ${id}_avnav/${id}_avnav_analysis.txt
python eval/analyze_scores.py ${id}_ana/${id}_ana.txt ${id}_ana/${id}_ana_analysis.txt
# python eval/analyze_scores.py ${id}_avna/${id}_avna.txt ${id}_avna/${id}_avna_analysis.txt
conda deactivate