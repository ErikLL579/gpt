#!/bin/bash
#SBATCH -A mp13
#SBATCH --job-name=obc5dlin60
#SBATCH --nodes=1
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=40:00:00
#SBATCH --output=slurm_obc5dlin60_%j.txt
#SBATCH --error=slurm_obc5dlin60_%j.txt

cd $SLURM_SUBMIT_DIR
module load python
source /global/cfs/cdirs/mp13/lundstrum/gpt-lundstrum-130726/gpt/lib/cgpt/build/source.sh

export OMP_NUM_THREADS=128

ROOT=$SCRATCH/nambu/obc5d_quad_60
mkdir -p $ROOT

python3 nambu_obc5d_topo_run.py --nsteps 70 --ntraj 20000 \
    --root $ROOT >> obc5d_quad_60.log 2>&1
