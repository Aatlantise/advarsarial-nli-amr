#!/bin/bash

#SBATCH --job-name="amr-graph"
#SBATCH --output="%x.o%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh
python bert-graph.py --seed 1 >> logs/amr-graph-1.log
python bert-graph.py --seed 2 >> logs/amr-graph-2.log
python bert-graph.py --seed 3 >> logs/amr-graph-3.log
