#!/bin/bash

#SBATCH --job-name="amr-only"
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
python bert.py --use_amr True --amr_only True --long True --seed 5 >> logs/amr-only-nospace-5-long.log
