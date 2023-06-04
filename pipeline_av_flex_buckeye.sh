#!/bin/bash                      
#SBATCH -t 6:00:00
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH -N 1
#SBATCH --exclude=node[100-116],node086,dgx001
#SBATCH --array 9

# use 80G for AV feature extraction for buckeye

id="be4620fyf_043023_001_"$SLURM_ARRAY_TASK_ID
corpora_audio="/om2/user/szhi/corpora/buckeye_segments_short"
corpora_audio_alignment="/om2/user/szhi/corpora/buckeye_segments_short"
corpora_video="/om2/user/szhi/corpora/buckeye_synthetic_video_short_new16"
data_pretrain="be4620fyf/pretrain.json"
data_train="be4620fyf/train.json"
data_test="be4620fyf/test.json"
video_fn="video_transform_7797"
offset=130
frame_fn="frame_transform_offset_frontpad_"$offset
video_n_components=4
dpgmm_alpha=0.01
# id="be1375fy_043023_"$SLURM_ARRAY_TASK_ID
# corpora_audio="/om2/user/szhi/corpora/buckeye_segments"
# corpora_audio_alignment="/om2/user/szhi/corpora/buckeye_segments"
# corpora_video="/om2/user/szhi/corpora/buckeye_synthetic_video_new16"
# data_pretrain="be1375fy/pretrain.json"
# data_train="be1375fy/train.json"
# data_test="be1375fy/test.json"
# video_fn="video_transform_7797"
# offset=130
# frame_fn="frame_transform_offset_frontpad_"$offset
# video_n_components=4
# dpgmm_alpha=1

echo $id
mkdir $id

### start wav2lip-local singularity image ###
source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles

cd /om2/user/szhi/perceptual-tuning-pnas

### switch to abx_eval conda env ###
source /om2/user/szhi/miniconda3/etc/profile.d/conda.sh
conda activate abx_eval

# # convert eval .npy to .features with mat2h5
# python tools/mat2h5.py $id/${id}_an_eval.npy $id/${id}_an_eval.features
# extract_posteriors 
rm $id/${id}_ava_flex_eval_posteriors.features
python tools/dpgmm/extract_posteriors.py $id/${id}_a_eval.features $id/${id}_1501-final.mat $id/${id}_ava_flex_eval_posteriors.features --flex_align left
# abxpy_eval
rm -r ${id}_ava_flex
mkdir ${id}_ava_flex
python eval/abxpy_eval_buckeye.py $data_test $id/${id}_ava_flex_eval_posteriors.features ${id}_ava_flex --av_offset $offset --alignment_corpus $corpora_audio_alignment

# # convert eval .npy to .features with mat2h5
# python tools/mat2h5.py $id/${id}_vn_eval.npy $id/${id}_vn_eval.features
# extract_posteriors 
# python tools/dpgmm/extract_posteriors.py $id/${id}_v_eval.features $id/${id}_1501-final.mat $id/${id}_avv_flex_eval_posteriors.features --flex_align right
# # abxpy_eval
# mkdir ${id}_avv_flex
# python eval/abxpy_eval_buckeye.py $data_test $id/${id}_avv_flex_eval_posteriors.features ${id}_avv_flex --av_offset $offset --alignment_corpus $corpora_audio_alignment

conda deactivate

### end abx_eval conda env ###

conda activate h5f_plt 
python eval/analyze_scores.py ${id}_ava_flex/${id}_ava_flex.txt ${id}_ava_flex/${id}_ava_flex_analysis.txt
# python eval/analyze_scores.py ${id}_avv_flex/${id}_avv_flex.txt ${id}_avv_flex/${id}_avv_flex_analysis.txt
conda deactivate