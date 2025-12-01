#!/usr/bin/env bash
set -euo pipefail

# Simple runner. Assumes current directory is the repo root.
export JULIA_NUM_THREADS="auto"

echo "-- Instantiating and precompiling project"
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "-- Generating figures"

# Planner plotting suite
julia --project --threads=auto scripts/plotting/planner/prelimineries.jl
julia --project --threads=auto scripts/plotting/planner/scc.jl
julia --project --threads=auto scripts/plotting/planner/optimal.jl
julia --project --threads=auto scripts/plotting/planner/discovery.jl
julia --project --threads=auto scripts/calibration/climate.jl

echo "-- All figure scripts completed"