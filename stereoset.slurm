#!/bin/bash
#SBATCH --job-name=stereoset_bert
#SBATCH --open-mode=append
#SBATCH --output=/home/bxw201/%j_%x.out
#SBATCH --error=/home/bxw201/%j_%x.err
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
cd /ext3/MLLU_Social-Bias/
pip install -r requirements.txt
cd StereoSet-master/code/
python eval_discriminative_models.py --pretrained-class 'roberta-base' --tokenizer RobertaTokenizer --intrasentence-model RoBERTaLM --intersentence-model RoBERTaLM --input-file ../data/dev.json --output-dir predictions/ --output-file predictions_roberta_RoBERTaLM_RoBERTaLM.json
python evaluation.py --gold-file ../data/dev.json --predictions-dir predictions/ --predictions-file predictions_roberta_RoBERTaLM_RoBERTaLM.json --output-file scores.json
"
