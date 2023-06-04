#!/bin/bash                      
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH --mem 80G
#SBATCH -N 1

id="be1375fy_oracle_brokenstops30mix"
corpora_audio="/om2/user/szhi/corpora/buckeye_segments"
corpora_audio_alignment="/om2/user/szhi/corpora/buckeye_segments"
corpora_video="/om2/user/szhi/corpora/buckeye_synthetic_video"
data_pretrain="be1375fy/pretrain.json"
data_train="be1375fy/train.json"
data_test="be1375fy/test.json"
video_fn="video_transform_7797"
offset=120
frame_fn="frame_transform_offset_frontpad_"$offset
video_n_components=20
dpgmm_alpha=1.0

echo $id
mkdir $id
cd /om2/user/szhi/perceptual-tuning-pnas

source /om2/user/szhi/miniconda3/etc/profile.d/conda.sh
conda activate synthetic_Dataset

# pretrain script
# singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/pretrain_visual_features.py $data_pretrain $id/$id.pkl --video_fn $video_fn --n_components $video_n_components --video_corpus $corpora_video
# extract audiovisual features for train and eval
# python video/extract_oracle_features_buckeye.py $data_train be1375fy_baseline/be1375fy_baseline.pkl $id/${id}_train.mat --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video
# python video/extract_oracle_features_buckeye.py $data_test be1375fy_baseline/be1375fy_baseline.pkl $id/${id}_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode av

# python video/extract_oracle_features_buckeye.py $data_train be1375fy_baseline/be1375fy_baseline.pkl $id/${id}_a_train.mat --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --mode a
# python video/extract_oracle_features_buckeye.py $data_test be1375fy_baseline/be1375fy_baseline.pkl $id/${id}_a_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode a 
# python video/extract_oracle_features_buckeye.py $data_train be1375fy_baseline/be1375fy_baseline.pkl $id/${id}_v_train.mat --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --mode v
# python video/extract_oracle_features_buckeye.py $data_test be1375fy_baseline/be1375fy_baseline.pkl $id/${id}_v_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode v
python video/extract_oracle_features_buckeye.py $data_test be1375fy_baseline/be1375fy_baseline.pkl $id/${id}_an_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode an
# python video/extract_oracle_features_buckeye.py $data_test be1375fy_baseline/be1375fy_baseline.pkl $id/${id}_vn_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode vn

conda deactivate

# # train AV dpgmm
# cp $id/${id}_train.mat /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/features/${id}_train.mat
# mkdir /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/$id

# module load mit/matlab/2020a
# cd /om2/user/szhi/perceptual-tuning-pnas/tools/dpgmm/dpmm_subclusters_2014-08-06/Gaussian/
# matlab -nojvm -r "feat_mat = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/features/${id}_train.mat'; model_dir = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/$id'; my_run_dpgmm_subclusters(feat_mat, model_dir, 15, true, $dpgmm_alpha); exit;"

# cd /om2/user/szhi/perceptual-tuning-pnas
# cp /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/$id/1501-final.mat $id/${id}_1501-final.mat

# # train A dpgmm
# cp $id/${id}_a_train.mat /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/features/${id}_a_train.mat
# mkdir /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/${id}_a

# module load mit/matlab/2020a
# cd /om2/user/szhi/perceptual-tuning-pnas/tools/dpgmm/dpmm_subclusters_2014-08-06/Gaussian/
# matlab -nojvm -r "feat_mat = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/features/${id}_a_train.mat'; model_dir = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/${id}_a'; my_run_dpgmm_subclusters(feat_mat, model_dir, 15, true, $dpgmm_alpha); exit;"

# cd /om2/user/szhi/perceptual-tuning-pnas
# cp /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/${id}_a/1501-final.mat $id/${id}_a_1501-final.mat

# # train V dpgmm
# cp $id/${id}_v_train.mat /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/features/${id}_v_train.mat
# mkdir /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/${id}_v

# module load mit/matlab/2020a
# cd /om2/user/szhi/perceptual-tuning-pnas/tools/dpgmm/dpmm_subclusters_2014-08-06/Gaussian/
# matlab -nojvm -r "feat_mat = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/features/${id}_v_train.mat'; model_dir = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/${id}_v'; my_run_dpgmm_subclusters(feat_mat, model_dir, 15, true, $dpgmm_alpha); exit;"

# cd /om2/user/szhi/perceptual-tuning-pnas
# cp /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/${id}_v/1501-final.mat $id/${id}_v_1501-final.mat


### switch to abx_eval conda env ###
conda activate abx_eval

# # convert eval .npy to .features with mat2h5
# python tools/mat2h5.py $id/${id}_eval.npy $id/${id}_eval.features
# # extract_posteriors 
# python tools/dpgmm/extract_posteriors.py $id/${id}_eval.features $id/${id}_1501-final.mat $id/${id}_eval_posteriors.features
# # abxpy_eval
# python eval/abxpy_eval_buckeye.py $data_test $id/${id}_eval_posteriors.features $id --av_offset $offset --alignment_corpus $corpora_audio_alignment

# mkdir ${id}_aa
# # convert eval .npy to .features with mat2h5
# python tools/mat2h5.py $id/${id}_a_eval.npy $id/${id}_a_eval.features
# # extract_posteriors 
# python tools/dpgmm/extract_posteriors.py $id/${id}_a_eval.features $id/${id}_a_1501-final.mat $id/${id}_aa_eval_posteriors.features
# # abxpy_eval
# python eval/abxpy_eval_buckeye.py $data_test $id/${id}_aa_eval_posteriors.features ${id}_aa --av_offset $offset --alignment_corpus $corpora_audio_alignment

# mkdir ${id}_vv
# # convert eval .npy to .features with mat2h5
# python tools/mat2h5.py $id/${id}_v_eval.npy $id/${id}_v_eval.features
# # extract_posteriors 
# python tools/dpgmm/extract_posteriors.py $id/${id}_v_eval.features $id/${id}_v_1501-final.mat $id/${id}_vv_eval_posteriors.features
# # abxpy_eval
# python eval/abxpy_eval_buckeye.py $data_test $id/${id}_vv_eval_posteriors.features ${id}_vv --av_offset $offset --alignment_corpus $corpora_audio_alignment

mkdir ${id}_ava
# convert eval .npy to .features with mat2h5
python tools/mat2h5.py $id/${id}_an_eval.npy $id/${id}_an_eval.features
# extract_posteriors 
python tools/dpgmm/extract_posteriors.py $id/${id}_an_eval.features $id/${id}_1501-final.mat $id/${id}_ava_eval_posteriors.features
# abxpy_eval
python eval/abxpy_eval_buckeye.py $data_test $id/${id}_ava_eval_posteriors.features ${id}_ava --av_offset $offset --alignment_corpus $corpora_audio_alignment

# mkdir ${id}_avv
# # convert eval .npy to .features with mat2h5
# python tools/mat2h5.py $id/${id}_vn_eval.npy $id/${id}_vn_eval.features
# # extract_posteriors 
# python tools/dpgmm/extract_posteriors.py $id/${id}_vn_eval.features $id/${id}_1501-final.mat $id/${id}_avv_eval_posteriors.features
# # abxpy_eval
# python eval/abxpy_eval_buckeye.py $data_test $id/${id}_avv_eval_posteriors.features ${id}_avv --av_offset $offset --alignment_corpus $corpora_audio_alignment

conda deactivate

conda activate h5f_plt 
# python eval/analyze_scores.py $id/${id}.txt $id/${id}_analysis.txt
# python eval/analyze_scores.py ${id}_aa/${id}_aa.txt ${id}_aa/${id}_aa_analysis.txt
# python eval/analyze_scores.py ${id}_vv/${id}_vv.txt ${id}_vv/${id}_vv_analysis.txt
python eval/analyze_scores.py ${id}_ava/${id}_ava.txt ${id}_ava/${id}_ava_analysis.txt
# python eval/analyze_scores.py ${id}_avv/${id}_avv.txt ${id}_avv/${id}_avv_analysis.txt

# python eval/analyze_clusters.py $id/${id}_eval_posteriors.features $data_test $id
# python eval/analyze_clusters.py $id/${id}_aa_eval_posteriors.features $data_test ${id}_aa
# python eval/analyze_clusters.py $id/${id}_vv_eval_posteriors.features $data_test ${id}_vv
python eval/analyze_clusters.py $id/${id}_ava_eval_posteriors.features $data_test ${id}_ava
# python eval/analyze_clusters.py $id/${id}_avv_eval_posteriors.features $data_test ${id}_avv

## ADD EVAL FOR AV-A+, AV-V+
### end abx_eval conda env ###