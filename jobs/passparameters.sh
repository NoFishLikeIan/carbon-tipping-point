#!/bin/bash
SLURM_ARRAY_TASK_ID=0

mkdir -p "$TMPDIR"/data # Create scratch directory with user name
rsync data/calibration.jld2 "$TMPDIR"/data # Copy calibration data to scratch

# Extract parameters at the SLURM_ARRAY_TASK_ID from parameters.json
jq_command='
.parameters[$id | tonumber] | 
"--rra \(.rra) --eis \(.eis) --threshold \(.threshold)" +
(if .leveldamages then " --leveldamages" else "" end) +
(if .allownegative then " --allownegative" else "" end)
'

params=$(jq -r --arg id "$SLURM_ARRAY_TASK_ID" "$jq_command" jobs/parameters.json)

# Run script
julia +1.9.2 --project scripts/runoptimal.jl \
    -d "data" -s "test-simulation" --verbose 1 --overwrite \
    -N 11 $params