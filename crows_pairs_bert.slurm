#!/bin/bash
#SBATCH --job-name=crows-pairs-bert
#SBATCH --open-mode=append
#SBATCH --output=/home/ma4759/%j_%x.out
#SBATCH --error=/home/ma4759/%j_%x.err
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4

## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.
singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
pip install -r requirements.txt
python src/crows-pairs.py --input_file data/crows_pairs_anonymized.csv --lm_model bert --output_file $SCRATCH/MLLU_model
"
