#!/bin/bash
#SBATCH -A mp13
#SBATCH --job-name=disloc60
#SBATCH --nodes=1
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=40:00:00
#SBATCH --output=slurm_disloc_60_%j.txt
#SBATCH --error=slurm_disloc_60_%j.txt

cd $SLURM_SUBMIT_DIR
module load python
source /global/cfs/cdirs/mp13/lundstrum/gpt-lundstrum-130726/gpt/lib/cgpt/build/source.sh

export OMP_NUM_THREADS=128

ROOT=$SCRATCH/nambu/disloc_60
mkdir -p $ROOT

# Seed from an equilibrated HMC checkpoint before FIRST launch (skips
# thermalization; use a different 7X000 per chain):
#   cp $SCRATCH/nambu/hmc_60/ckpoint_lat.74000 $ROOT/ckpoint_lat.0
# Verify plaq ~ 0.594, t2E ~ 0.10-0.17 on the first measurement.
#
# gamma=1 idle => dH should match HMC's at tau=2/nsteps=50; if attempt
# trajectories (elevated |D|) are selectively rejected, shorten --tau
# first. lam tuning: watch "stail" (equilibrium ~0.08; attempts should
# push armed% > 0) — raise --lam if attempts never arm.

python3 nambu_disloc_topo_run.py --beta 6.0 \
    --gamma 1.0 --c_p 0.75 --c_r 1.5 --lam 1.75 \
    --nsm1 1 --nsm2 3 \
    --tau 2.0 --nsteps 50 --ntraj 20000 \
    --seed nambu-disloc \
    --root $ROOT >> disloc_60.log 2>&1
