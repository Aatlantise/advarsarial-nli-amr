#!/bin/bash
#!/bin/bash

#SBATCH --job-name="nli-bert"
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
python bert.py --seed 1 --use_amr False >> baseline_1.log
python bert.py --seed 2 --use_amr False >> baseline_2.log
python bert.py --seed 3 --use_amr False >> baseline_3.log
