#!/bin/bash
#SBATCH --job-name=crows-pairs-bert
#SBATCH --open-mode=append
#SBATCH --output=/home/$USER/%j_%x.out
#SBATCH --error=/home/$USER/%j_%x.err
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
pip install -r requirements.txt
cd StereoSet-master/code/
python eval_discriminative_models.py --pretrained-class bert-base-uncased --tokenizer BertTokenizer --intrasentence-model BertLM --intersentence-model BertNextSentence --input-file ../data/dev.json --output-dir predictions/
"
