#!/bin/bash
#SBATCH --job-name=agent_gnn
#SBATCH --output=/mnt/home/dkoehler/all_projects/pandapower-env/scripts/out/test_greedy__%j.log
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40gb
#SBATCH --time=7-20:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --partition=kes
#SBATCH --mail-type=FAIL

# Ensure virtual environment is set up and dependencies are installed
poetry install --no-interaction --no-root
# Run the target script inside the poetry-managed environment
poetry run python scripts/run_greedy.py