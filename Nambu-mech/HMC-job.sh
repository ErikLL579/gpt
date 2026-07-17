#!/bin/bash
#SBATCH -A mp13
#SBATCH --job-name=hmc60
#SBATCH --nodes=1
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --output=slurm_hmc60_%j.txt
#SBATCH --error=slurm_hmc60_%j.txt

cd $SLURM_SUBMIT_DIR          # submit from Nambu-mech/

module load python
source /global/cfs/cdirs/mp13/lundstrum/gpt-lundstrum-130726/gpt/lib/cgpt/build/source.sh
export OMP_NUM_THREADS=128    # all physical cores

ROOT=$SCRATCH/nambu/hmc_60
mkdir -p $ROOT

python3 hmc_topo_run.py --beta 6.0 --nsteps 50 --root $ROOT >> hmc_60.log 2>&1

ROOT=$SCRATCH/nambu/obc5d_60
python3 nambu_obc5d_topo_run.py --beta 6.0 --nsteps 50 --ntraj 20000 --root $ROOT >> obc5d_60.log 2>&1
