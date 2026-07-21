#!/bin/bash
#SBATCH -A mp13
#SBATCH --job-name=disloc160
#SBATCH --nodes=1
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=40:00:00
#SBATCH --output=slurm_disloc_160_%j.txt
#SBATCH --error=slurm_disloc_160_%j.txt

cd $SLURM_SUBMIT_DIR
module load python
source /global/cfs/cdirs/mp13/lundstrum/gpt-lundstrum-130726/gpt/lib/cgpt/build/source.sh

export OMP_NUM_THREADS=128

ROOT=$SCRATCH/nambu/disloc_160
mkdir -p $ROOT

# Seed from an equilibrated 16^4 HMC checkpoint before FIRST launch
# (skips thermalization; every measurement is then an equilibrium sample):
#   cp $SCRATCH/nambu/hmc_160/ckpoint_lat.XXXX $ROOT/ckpoint_lat.0
# Verify plaq ~ 0.594 and a sane t2E on the first measurement.
#
# lam: 8^4 calibration was lam=1.75 (equilibrium |s| p99.9 ~ 0.1). D is a
# global sum whose equilibrium floor grows ~ sqrt(V) (Q fluctuations), so at
# 16x the volume expect stail ~4x at fixed lam => start at lam = 1.75/4 ~ 0.45
# and RETUNE from the per-trajectory "stail" vital: target p99.9 |s| ~ 0.1,
# i.e. ~7x below the window edge c_p = 0.75. armed% should be ~0 in
# equilibrium and light up only on elevated-|D| trajectories.
#
# nsteps 100 matches the 16^4 HMC baseline (gamma=1 idle => same dH scaling).
# THE diagnostic: acceptance conditioned on elevated |D| — selective
# rejection of attempt trajectories silently vetoes crossings; shorten --tau
# first if seen (armed Lyapunov growth is exponential in tau).
#
# Cost: AD stout-chain force scales ~V; expect O(2-3 min)/traj => 20k is a
# multi-week chain of 40h legs. Auto-resume + afterany dependencies:
#   j=$(sbatch --parsable disloc_160.sbatch); sbatch --dependency=afterany:$j disloc_160.sbatch
# Judge per-trajectory (tau_int) before burning the full budget.

python3 nambu_disloc_topo_run.py --beta 6.0 --L 16 \
    --gamma 1.0 --c_p 0.75 --c_r 1.5 --lam 0.45 \
    --nsm1 1 --nsm2 3 \
    --tau 2.0 --nsteps 100 --ntraj 20000 \
    --seed nambu-disloc-160 \
    --root $ROOT >> disloc_160.log 2>&1
