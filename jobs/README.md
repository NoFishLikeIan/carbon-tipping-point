# Jobs

Some bash and julia scripts to speed up multiple versioning. This is needed to run scripts on Snellius since I locally use `julia 1.10` and remotely `julia 1.7`.

1. `switchversions.jl` allows to switch between the local and snellius branches.
2. `syncfiles.sh` syncs the file needed on Snellius to run the simulation.
3. `simulate.job` runs the simulation on Snellius using the `.env` variables.

## Time estimation

