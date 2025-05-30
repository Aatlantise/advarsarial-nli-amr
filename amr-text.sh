#!/bin/bash

#SBATCH --job-name="amr-text"
#SBATCH --output="%x.o%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh
python bert.py --use_amr True --amr_only False --seed 1 >> logs/amr-text-1.log
python bert.py --use_amr True --amr_only False --seed 2 >> logs/amr-text-2.log
python bert.py --use_amr True --amr_only False --seed 3 >> logs/amr-text-3.log