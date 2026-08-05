#!/bin/bash
#SBATCH -A mp13
#SBATCH --job-name=hmc60
#SBATCH --nodes=1
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=40:00:00
#SBATCH --output=slurm_hmc_60_%j.txt
#SBATCH --error=slurm_hmc_60_%j.txt

cd $SLURM_SUBMIT_DIR
module load python
source /global/cfs/cdirs/mp13/lundstrum/gpt-lundstrum-130726/gpt/lib/cgpt/build/source.sh

export OMP_NUM_THREADS=128

ROOT=$SCRATCH/nambu/hmc_60
mkdir -p $ROOT

# 8^4 pure SU(3) gauge theory baseline (Wilson action, plain HMC leapfrog,
# Metropolis; Wilson-flowed Q5LI/Qclover + energy density measured every
# trajectory). Fresh cold start; --ntherm 500 covers the slow IR (t2E)
# equilibration (~10x slower than plaquette).
# Benchmark at these params: ~1.1 s/traj, acceptance ~0.80 (July 2026).
#
# --autotune 1: nsteps stays fixed at 50 (eps = tau/nsteps); tau is
# adjusted every 50 trajectories over the first 300 post-therm
# trajectories so the window acceptance lands in [0.80, 0.90], then
# frozen for the remaining ~19700 trajectories. tau/tuning state is
# persisted to $ROOT/tau_state.txt, so this is safe to submit once with
# a large --ntraj and forget about acceptance tuning entirely.
#
# Auto-resume makes walltime kills harmless:
#   j=$(sbatch --parsable hmc_60.sbatch); sbatch --dependency=afterany:$j hmc_60.sbatch
# Once equilibrated, its ckpoint_lat.N files seed the disloc/amplifier chains.

python3 hmc_topo_run.py --beta 6.0 --L 8 \
    --tau 2.0 --nsteps 50 --ntherm 500 --ntraj 20000 \
    --autotune 1 --tune-window 50 --tune-ntraj 300 --tune-lo 0.80 --tune-hi 0.90 \
    --seed hmc-topo-60 \
    --root $ROOT >> hmc_60.log 2>&1
