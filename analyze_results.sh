#!/bin/bash                      
#SBATCH -t 5:00:00
#SBATCH -n 1
#SBATCH --array 1-8
#SBATCH --exclude=node[100-116],node086

id="beminifyf_vdim"$SLURM_ARRAY_TASK_ID
eval_json='beminifyf/test.json'
alignment_corpus='/om2/user/szhi/corpora/buckeye_segments'
offset=120

echo $id
echo $eval_json
echo $alignment_corpus
echo $offset

source /om2/user/szhi/miniconda3/etc/profile.d/conda.sh
conda activate abx_eval

# rm ${id}_avav/*.abx
# rm ${id}_avav/*.distance
# rm ${id}_avav/*.scores
# rm ${id}_avav/*.txt
# rm ${id}_ava/*.abx
# rm ${id}_ava/*.distance
# rm ${id}_ava/*.scores
# rm ${id}_ava/*.txt
# rm ${id}_avv/*.abx
# rm ${id}_avv/*.distance
# rm ${id}_avv/*.scores
# rm ${id}_avv/*.txt
# rm ${id}_aa/*.abx
# rm ${id}_aa/*.distance
# rm ${id}_aa/*.scores
# rm ${id}_aa/*.txt
# rm ${id}_vv/*.abx
# rm ${id}_vv/*.distance
# rm ${id}_vv/*.scores
# rm ${id}_vv/*.txt
# rm ${id}_vv/*.item

for i in 0 1 2 3 4
do
    rm ${id}_${i}_vv/*.abx
    rm ${id}_${i}_vv/*.distance
    rm ${id}_${i}_vv/*.scores
    rm ${id}_${i}_vv/*.txt
    rm ${id}_${i}_vv/*.item
    python eval/abxpy_eval_buckeye.py $eval_json ${id}_${i}/${id}_${i}_vv_eval_posteriors.features ${id}_${i}_vv --av_offset $offset --alignment_corpus $alignment_corpus
done

# python eval/abxpy_eval_buckeye.py $eval_json $id/${id}_avav_eval_posteriors.features ${id}_avav --av_offset $offset --alignment_corpus $alignment_corpus
# python eval/abxpy_eval_buckeye.py $eval_json $id/${id}_ava_eval_posteriors.features ${id}_ava --av_offset $offset --alignment_corpus $alignment_corpus
# python eval/abxpy_eval_buckeye.py $eval_json $id/${id}_avv_eval_posteriors.features ${id}_avv --av_offset $offset --alignment_corpus $alignment_corpus
# python eval/abxpy_eval_buckeye.py $eval_json $id/${id}_aa_eval_posteriors.features ${id}_aa --av_offset $offset --alignment_corpus $alignment_corpus
# python eval/abxpy_eval_buckeye.py $eval_json $id_${i}/${id}_${i}_vv_eval_posteriors.features ${id}_${i}_vv --av_offset $offset --alignment_corpus $alignment_corpus

# rm ${id}_avam/*.abx
# rm ${id}_avam/*.distance
# rm ${id}_avam/*.scores
# rm ${id}_avam/*.txt
# rm ${id}_avvm/*.abx
# rm ${id}_avvm/*.distance
# rm ${id}_avvm/*.scores
# rm ${id}_avvm/*.txt
# python eval/abxpy_eval_buckeye.py $eval_json $id/${id}_avam_eval_posteriors.features ${id}_avam --av_offset $offset --alignment_corpus $alignment_corpus
# python eval/abxpy_eval_buckeye.py $eval_json $id/${id}_avvm_eval_posteriors.features ${id}_avvm --av_offset $offset --alignment_corpus $alignment_corpus

conda deactivate
conda activate h5f_plt 

# python eval/analyze_clusters.py ${id}/${id}_avav_eval_posteriors.features $eval_json ${id}_avav --alignment_corpus $alignment_corpus
# python eval/analyze_clusters.py ${id}/${id}_ava_eval_posteriors.features $eval_json ${id}_ava --alignment_corpus $alignment_corpus
# python eval/analyze_clusters.py ${id}/${id}_avv_eval_posteriors.features $eval_json ${id}_avv --alignment_corpus $alignment_corpus
# python eval/analyze_clusters.py ${id}/${id}_aa_eval_posteriors.features $eval_json ${id}_aa --alignment_corpus $alignment_corpus
# python eval/analyze_clusters.py ${id}/${id}_vv_eval_posteriors.features $eval_json ${id}_vv --alignment_corpus $alignment_corpus

# python eval/analyze_scores.py ${id}_avav/${id}_avav.txt ${id}_avav/${id}_avav_analysis.txt
# python eval/analyze_scores.py ${id}_ava/${id}_ava.txt ${id}_ava/${id}_ava_analysis.txt
# python eval/analyze_scores.py ${id}_avv/${id}_avv.txt ${id}_avv/${id}_avv_analysis.txt
# python eval/analyze_scores.py ${id}_aa/${id}_aa.txt ${id}_aa/${id}_aa_analysis.txt
for i in 0 1 2 3 4
do
    python eval/analyze_scores.py ${id}_${i}_vv/${id}_${i}_vv.txt ${id}_${i}_vv/${id}_${i}_vv_analysis.txt
done

# python eval/analyze_scores.py ${id}_avam/${id}_avam.txt ${id}_avam/${id}_avam_analysis.txt
# python eval/analyze_scores.py ${id}_avvm/${id}_avvm.txt ${id}_avvm/${id}_avvm_analysis.txt

# python eval/analyze_scores.py eval_first_window_fix/eval_first_window_fix.txt eval_first_window_fix/eval_first_window_fix_analysis.txt
# python eval/analyze_scores.py eval_second_window/eval_second_window.txt eval_second_window/eval_second_window_analysis.txt
# python eval/analyze_scores.py eval_last_window/eval_last_window.txt eval_last_window/eval_last_window_analysis.txt
# python eval/analyze_scores.py eval_lastfull_window/eval_lastfull_window.txt eval_lastfull_window/eval_lastfull_window_analysis.txt

conda deactivate