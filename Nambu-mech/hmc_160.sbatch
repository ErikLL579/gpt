#!/bin/bash
#SBATCH -A mp13
#SBATCH --job-name=hmc160
#SBATCH --nodes=1
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=40:00:00
#SBATCH --output=slurm_hmc_160_%j.txt
#SBATCH --error=slurm_hmc_160_%j.txt

cd $SLURM_SUBMIT_DIR
module load python
source /global/cfs/cdirs/mp13/lundstrum/gpt-lundstrum-130726/gpt/lib/cgpt/build/source.sh

export OMP_NUM_THREADS=128

ROOT=$SCRATCH/nambu/hmc_160
mkdir -p $ROOT

# 16^4 baseline. Fresh start (no equilibrated 16^4 checkpoint exists yet):
# --ntherm 500 covers the slow IR (t2E) equilibration, ~10x slower than plaq.
# nsteps 100 = 2x the 8^4 value (dH ~ eps^4 V at fixed tau => eps ~ V^-1/4);
# check acceptance lands in 65-85% on the first few hundred trajectories.
# Expect ~10-20 s/traj => 20k needs 2-3 chained legs:
#   j=$(sbatch --parsable hmc_160.sbatch); sbatch --dependency=afterany:$j hmc_160.sbatch
# Auto-resume makes walltime kills harmless.
# Once equilibrated, its ckpoint_lat.N files seed the disloc/amplifier chains.

python3 hmc_topo_run.py --beta 6.0 --L 16 \
    --tau 2.0 --nsteps 100 --ntherm 500 --ntraj 20000 \
    --seed hmc-topo-160 \
    --root $ROOT >> hmc_160.log 2>&1
