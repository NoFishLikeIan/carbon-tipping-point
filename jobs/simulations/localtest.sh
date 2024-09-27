# Synthetic SLURM parameters
SLURM_ARRAY_TASK_ID=0
SLURM_CPUS_PER_TASK="auto"


mkdir -p "$TMPDIR"/data # Create scratch directory with user name
rsync data/calibration.jld2 "$TMPDIR"/data # Copy calibration data to scratch

# Extract parameters at the SLURM_ARRAY_TASK_ID from parameters.json
jq_command='
.benchmarkparameters[$id | tonumber] | 
"--rra \(.rra) --eis \(.eis)" +
(if .leveldamages then " --leveldamages" else "" end) +
(if .allownegative then " --allownegative" else "" end)
'

params=$(jq -r --arg id "$SLURM_ARRAY_TASK_ID" "$jq_command" jobs/simulations/parameters.json)

# Run script
julia +1.9.2 --threads ${SLURM_CPUS_PER_TASK} --project scripts/runbenchmark.jl \
    -d "data" -s "simulation-test" --verbose 2 --overwrite \
    -N 21 $params

# Copy results back to home
rsync -avzu "$TMPDIR"/data $HOME/scc-tipping-points