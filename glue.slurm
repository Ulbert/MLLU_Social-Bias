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
export TASK_NAME=mrpc

## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.
singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
pip install -r src/requirements.txt
python run_glue.py \\
  --model_name_or_path bert-base-uncased \\
  --task_name $TASK_NAME \\
  --do_train \\
  --do_eval \\
  --max_seq_length 128 \\
  --per_device_train_batch_size 32 \\
  --learning_rate 2e-5 \\
  --num_train_epochs 3 \\
  --output_dir /$SCRATCH/glue/$TASK_NAME
"
