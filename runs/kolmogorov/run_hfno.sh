#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH -J kolmogorov34
#SBATCH -o /scratch/cayuelam/logs/kolmogorov/%x_%j.out

module purge
module load python/3.11  

source activate fto
pip install -r requirements.txt

rsync -av $STORE/data/kolmogorov/ $SCRATCH/data/kolmogorov

export DATA_DIR=$SCRATCH/data/
export LOG_DIR=$SCRATCH/fno/runs/

python main_diffusion.py 
source deactivate

