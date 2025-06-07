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
python bert.py --use_amr True --seed 1 >> logs/amr-text-nospace-1.log
python bert.py --use_amr True --seed 2 >> logs/amr-text-nospace-2.log
python bert.py --use_amr True --seed 3 >> logs/amr-text-nospace-3.log