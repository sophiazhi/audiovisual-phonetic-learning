#!/bin/bash                      
#SBATCH -t 2:00:00
#SBATCH -n 1

# id="hmm_test"
id="be1375fy_dpgmm_120_ss"
eval_json='be1375fy/test.json'
alignment_corpus='/om2/user/szhi/corpora/buckeye_segments'

source /om2/user/szhi/miniconda3/etc/profile.d/conda.sh
conda activate h5f_plt 

python eval/analyze_clusters.py ${id}/${id}_avav_eval_posteriors.features $eval_json ${id}_avav --alignment_corpus $alignment_corpus
python eval/analyze_clusters.py ${id}/${id}_ava_eval_posteriors.features $eval_json ${id}_ava --alignment_corpus $alignment_corpus
python eval/analyze_clusters.py ${id}/${id}_avv_eval_posteriors.features $eval_json ${id}_avv --alignment_corpus $alignment_corpus
python eval/analyze_clusters.py ${id}/${id}_aa_eval_posteriors.features $eval_json ${id}_aa --alignment_corpus $alignment_corpus
python eval/analyze_clusters.py ${id}/${id}_vv_eval_posteriors.features $eval_json ${id}_vv --alignment_corpus $alignment_corpus

# python eval/analyze_scores.py $id/${id}.txt ${id}/${id}_analysis.txt
# python eval/analyze_scores.py ${id}_ava/${id}_ava.txt ${id}_ava/${id}_ava_analysis.txt
# python eval/analyze_scores.py ${id}_avv/${id}_avv.txt ${id}_avv/${id}_avv_analysis.txt
# python eval/analyze_scores.py ${id}_aa/${id}_aa.txt ${id}_aa/${id}_aa_analysis.txt
# python eval/analyze_scores.py ${id}_vv/${id}_vv.txt ${id}_vv/${id}_vv_analysis.txt

conda deactivate