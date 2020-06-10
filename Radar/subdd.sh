#!/bin/bash
#SBATCH --job-name=radardd
#SBATCH -p serc
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --time=08:00:00
#SBATCH --mail-user=bdelorme@stanford.edu
#SBATCH --mail-type=END

python run_all_ddnet.py
