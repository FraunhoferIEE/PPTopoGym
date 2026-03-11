#!/bin/bash
#SBATCH --job-name=agent_gnn
#SBATCH --output=/PATH/TO/ray_gnn%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40gb
#SBATCH --time=180:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=kes
#SBATCH --mail-type=FAIL

date;hostname;pwd

source ~/.bashrc
conda activate pandaenv

srun python -u /PATH/TO/gnn_training_example_script.py
