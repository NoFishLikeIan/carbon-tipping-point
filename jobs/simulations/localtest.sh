# Synthetic SLURM parameters
SLURM_CPUS_PER_TASK="auto"

SCRATCHDATA="$TMPDIR"/data

mkdir -p $SCRATCHDATA # Create scratch directory with user name
rsync data/calibration.jld2 $SCRATCHDATA # Copy calibration data to scratch

# Run script
julia +1.9.2 --threads ${SLURM_CPUS_PER_TASK} --project scripts/runtipping.jl -N 31 \
    --verbose 2 --cachestep 0.25 \
    --datapath "$SCRATCHDATA"  --simulationpath "simulation-local-small" --overwrite

# Copy results back to home
rsync -avzu $SCRATCHDATA $HOME/economics/scc-tipping-points
rm -r $SCRATCHDATA