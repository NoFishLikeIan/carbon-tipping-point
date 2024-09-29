# Synthetic SLURM parameters
SLURM_CPUS_PER_TASK="auto"

mkdir -p "$TMPDIR"/data # Create scratch directory with user name
rsync data/calibration.jld2 "$TMPDIR"/data # Copy calibration data to scratch

# Run script
julia +1.9.2 --threads ${SLURM_CPUS_PER_TASK} \
    --project scripts/runtipping.jl \
    -d "data" -s "simulation-local" --verbose 1 --overwrite \
    -N 41 --cachestep 0.25

# Copy results back to home
rsync -avzu "$TMPDIR"/data $HOME/scc-tipping-points