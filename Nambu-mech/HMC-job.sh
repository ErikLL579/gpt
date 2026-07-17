#!/bin/bash
#SBATCH -A mp13
#SBATCH --job-name=hmc60
#SBATCH --nodes=1
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=30:00:00
#SBATCH --output=slurm_hmc60_%j.txt
#SBATCH --error=slurm_hmc60_%j.txt

cd $SLURM_SUBMIT_DIR          # submit from Nambu-mech/

module load python
source /global/cfs/cdirs/mp13/lundstrum/gpt-lundstrum-130726/gpt/lib/cgpt/build/source.sh
export OMP_NUM_THREADS=128    # all physical cores

ROOT=$SCRATCH/nambu/hmc_65
mkdir -p $ROOT

python3 hmc_topo_run.py --beta 6.5 --L 8 \
      --tau 2.0 --nsteps 50 --ntherm 100 --ntraj 20000 \
      --seed hmc-8-65 \
      --root $SCRATCH/nambu/hmc_84_65

