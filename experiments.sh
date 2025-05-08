#!/bin/bash

#SBATCH --job-name="nli-amr-bert"
#SBATCH --output="%x.o%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh

python bert.py --use_amr=False
python bert.py --use_amr=True
