#!/bin/bash
#SBATCH -A mp13
#SBATCH --job-name=amplaggr60
#SBATCH --nodes=1
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=40:00:00
#SBATCH --output=slurm_ampl_aggr_60_%j.txt
#SBATCH --error=slurm_ampl_aggr_60_%j.txt

cd $SLURM_SUBMIT_DIR
module load python
source /global/cfs/cdirs/mp13/lundstrum/gpt-lundstrum-130726/gpt/lib/cgpt/build/source.sh

export OMP_NUM_THREADS=128

ROOT=$SCRATCH/nambu/ampl_aggr_60
mkdir -p $ROOT

python3 nambu_ampl_topo_run.py --beta 6.0 \
    --c_p 0.581 --c_r 1.936 --lam 1.453 \
    --tau 1.0 --nsteps 50 --ntraj 20000 \
    --seed ampl-aggr \
    --root $ROOT >> ampl_aggr_60.log 2>&1
