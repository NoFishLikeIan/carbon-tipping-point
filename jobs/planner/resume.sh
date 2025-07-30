FILEPATH=$0 # Assumes the first argument passed in the location of the file

# Run script
julia +1.9.2 --threads ${SLURM_CPUS_PER_TASK} \
    --project scripts/runtipping.jl \
    -d "data" -s "simulation-local" --verbose 2 --overwrite \
    -N 31