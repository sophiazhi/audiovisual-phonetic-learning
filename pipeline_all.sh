#!/bin/bash                      
#SBATCH -t 36:00:00
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

echo $id
mkdir $id

### start wav2lip-local singularity image ###
source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles

cd /om2/user/szhi/perceptual-tuning-pnas

### extract features
# pretrain script
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/pretrain_visual_features_buckeye.py $data_pretrain $id/$id.pkl --video_fn $video_fn --n_components $video_n_components --video_corpus $corpora_video
# extract train features
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features_buckeye.py $data_train $id/$id.pkl $id/${id}_av_train.mat --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features_buckeye.py $data_train $id/$id.pkl $id/${id}_a_train.mat --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --mode a
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features_buckeye.py $data_train $id/$id.pkl $id/${id}_v_train.mat --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --mode v
# extract eval features
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features_buckeye.py $data_test $id/$id.pkl $id/${id}_av_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode av
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features_buckeye.py $data_test $id/$id.pkl $id/${id}_a_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode a
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features_buckeye.py $data_test $id/$id.pkl $id/${id}_v_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode v
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features_buckeye.py $data_test $id/$id.pkl $id/${id}_nv_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode nv
singularity exec --nv --bind /om2/user/szhi/ /om2/user/szhi/vagrant/wav2lip-local.simg python video/extract_audiovisual_features_buckeye.py $data_test $id/$id.pkl $id/${id}_n_eval.npy --video_fn $video_fn --frame_fn $frame_fn --audio_corpus $corpora_audio --video_corpus $corpora_video --split eval --mode n


### train dpgmms
module load mit/matlab/2020a
# train AV dpgmm
cp $id/${id}_av_train.mat /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/features/${id}_av_train.mat
mkdir /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/models/${id}_av
cd /om2/user/szhi/perceptual-tuning-pnas/tools/dpgmm/dpmm_subclusters_2014-08-06/Gaussian/
matlab -nojvm -r "feat_mat = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/features/${id}_av_train.mat'; model_dir = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/models/${id}_av'; my_run_dpgmm_subclusters(feat_mat, model_dir, 15, true, $dpgmm_alpha); exit;"
cd /om2/user/szhi/perceptual-tuning-pnas
cp /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/models/${id}_av/1501-final.mat $id/${id}_av_1501-final.mat
# train A dpgmm
cp $id/${id}_a_train.mat /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/features/${id}_a_train.mat
mkdir /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/models/${id}_a
cd /om2/user/szhi/perceptual-tuning-pnas/tools/dpgmm/dpmm_subclusters_2014-08-06/Gaussian/
matlab -nojvm -r "feat_mat = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/features/${id}_a_train.mat'; model_dir = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/models/${id}_a'; my_run_dpgmm_subclusters(feat_mat, model_dir, 15, true, $dpgmm_alpha); exit;"
cd /om2/user/szhi/perceptual-tuning-pnas
cp /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/models/${id}_a/1501-final.mat $id/${id}_a_1501-final.mat
# train V dpgmm
cp $id/${id}_v_train.mat /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/features/${id}_v_train.mat
mkdir /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/models/${id}_v
cd /om2/user/szhi/perceptual-tuning-pnas/tools/dpgmm/dpmm_subclusters_2014-08-06/Gaussian/
matlab -nojvm -r "feat_mat = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/features/${id}_v_train.mat'; model_dir = '/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/models/${id}_v'; my_run_dpgmm_subclusters(feat_mat, model_dir, 15, true, $dpgmm_alpha); exit;"
cd /om2/user/szhi/perceptual-tuning-pnas
cp /om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/Buckeye/models/${id}_v/1501-final.mat $id/${id}_v_1501-final.mat

### switch to abx_eval conda env ###
source /om2/user/szhi/miniconda3/etc/profile.d/conda.sh
conda activate abx_eval

python tools/mat2h5.py $id/${id}_av_eval.npy $id/${id}_av_eval.features
python tools/mat2h5.py $id/${id}_a_eval.npy $id/${id}_a_eval.features
python tools/mat2h5.py $id/${id}_v_eval.npy $id/${id}_v_eval.features
python tools/mat2h5.py $id/${id}_nv_eval.npy $id/${id}_nv_eval.features
python tools/mat2h5.py $id/${id}_n_eval.npy $id/${id}_n_eval.features

python tools/dpgmm/extract_posteriors.py $id/${id}_av_eval.features $id/${id}_av_1501-final.mat $id/${id}_avav_eval_posteriors.features
mkdir ${id}_avav
python eval/abxpy_eval_buckeye.py $data_test $id/${id}_avav_eval_posteriors.features ${id}_avav --av_offset $offset --alignment_corpus $corpora_audio_alignment

python tools/dpgmm/extract_posteriors.py $id/${id}_a_eval.features $id/${id}_av_1501-final.mat $id/${id}_ava_eval_posteriors.features --flex_align left
mkdir ${id}_ava
python eval/abxpy_eval_buckeye.py $data_test $id/${id}_ava_eval_posteriors.features ${id}_ava --av_offset $offset --alignment_corpus $corpora_audio_alignment

python tools/dpgmm/extract_posteriors.py $id/${id}_v_eval.features $id/${id}_av_1501-final.mat $id/${id}_avv_eval_posteriors.features --flex_align right
mkdir ${id}_avv
python eval/abxpy_eval_buckeye.py $data_test $id/${id}_avv_eval_posteriors.features ${id}_avv --av_offset $offset --alignment_corpus $corpora_audio_alignment

python tools/dpgmm/extract_posteriors.py $id/${id}_a_eval.features $id/${id}_a_1501-final.mat $id/${id}_aa_eval_posteriors.features
mkdir ${id}_aa
python eval/abxpy_eval_buckeye.py $data_test $id/${id}_aa_eval_posteriors.features ${id}_aa --av_offset $offset --alignment_corpus $corpora_audio_alignment

python tools/dpgmm/extract_posteriors.py $id/${id}_v_eval.features $id/${id}_v_1501-final.mat $id/${id}_vv_eval_posteriors.features
mkdir ${id}_vv
python eval/abxpy_eval_buckeye.py $data_test $id/${id}_vv_eval_posteriors.features ${id}_vv --av_offset $offset --alignment_corpus $corpora_audio_alignment

python tools/dpgmm/extract_posteriors.py $id/${id}_nv_eval.features $id/${id}_av_1501-final.mat $id/${id}_avnv_eval_posteriors.features 
mkdir ${id}_avnv
python eval/abxpy_eval_buckeye.py $data_test $id/${id}_avnv_eval_posteriors.features ${id}_avnv --av_offset $offset --alignment_corpus $corpora_audio_alignment

python tools/dpgmm/extract_posteriors.py $id/${id}_n_eval.features $id/${id}_a_1501-final.mat $id/${id}_an_eval_posteriors.features
mkdir ${id}_an
python eval/abxpy_eval_buckeye.py $data_test $id/${id}_an_eval_posteriors.features ${id}_an --av_offset $offset --alignment_corpus $corpora_audio_alignment

conda deactivate

### end abx_eval conda env ###

conda activate h5f_plt 
python eval/analyze_scores.py ${id}_avav/${id}_avav.txt ${id}_avav/${id}_avav_analysis.txt
python eval/analyze_scores.py ${id}_ava/${id}_ava.txt ${id}_ava/${id}_ava_analysis.txt
python eval/analyze_scores.py ${id}_avv/${id}_avv.txt ${id}_avv/${id}_avv_analysis.txt
python eval/analyze_scores.py ${id}_aa/${id}_aa.txt ${id}_aa/${id}_aa_analysis.txt
python eval/analyze_scores.py ${id}_vv/${id}_vv.txt ${id}_vv/${id}_vv_analysis.txt
python eval/analyze_scores.py ${id}_avnv/${id}_avnv.txt ${id}_avnv/${id}_avnv_analysis.txt
python eval/analyze_scores.py ${id}_an/${id}_an.txt ${id}_an/${id}_an_analysis.txt
conda deactivate