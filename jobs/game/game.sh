#!/bin/bash

#SBATCH --job-name=game
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --partition=rome
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH -o ./logs/out/GAME%A_%a.out 
#SBATCH -e ./logs/error/GAME%A_%a.out
#SBATCH --mail-user=a.titton@uva.nl

# There are 2 parameters set
module load 2023
module load Julia/1.9.2-linux-x86_64

cd $HOME/scc-tipping-points # Move to base directory

SCRATCHDATA="$TMPDIR"/data
mkdir -p $SCRATCHDATA # Create scratch directory with user name
rsync data/calibration.jld2 $SCRATCHDATA # Copy calibration data to scratch

# Extract parameters at the SLURM_ARRAY_TASK_ID from parameters.json
jq_command='
.tippingparameters[$id | tonumber] | 
"--rra \(.rra) --eis \(.eis) --threshold \(.threshold)" +
(if .leveldamages then " --leveldamages" else "" end) +
(if .allownegative then " --allownegative" else "" end)
'

params=$(jq -r --arg id "$SLURM_ARRAY_TASK_ID" "$jq_command" jobs/simulations/parameters.json)

# Run script 
julia --threads ${SLURM_CPUS_PER_TASK} --project scripts/rungame.jl -N 71 \
    --cachestep 0.25 --verbose 1 \
    --datapath "$SCRATCHDATA"  --simulationpath "game-simulation-large" --overwrite \
    $params

# Copy results back to home
rsync -avzu $SCRATCHDATA $HOME/scc-tipping-points
rm -r $SCRATCHDATA # Remove scratch directory