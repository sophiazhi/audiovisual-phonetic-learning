#!/bin/bash                      
#SBATCH -t 48:00:00
#SBATCH -n 10

train_mat="/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/features/split5500_train.mat"
model_dir="/om2/user/szhi/perceptual-tuning-pnas/models/dpgmm/CHILDES/models/split5500_alpha0-1"
alpha=0.1

mkdir $model_dir
module load mit/matlab/2020a
cd /om2/user/szhi/perceptual-tuning-pnas/tools/dpgmm/dpmm_subclusters_2014-08-06/Gaussian/
matlab -nojvm -r "feat_mat = '$train_mat'; model_dir = '$model_dir'; my_run_dpgmm_subclusters(feat_mat, model_dir, 15, true, $alpha); exit;"
