# Jobs

Some bash and julia scripts to speed up multiple versioning. This is needed to run scripts on Snellius since I locally use `julia 1.10` and remotely `julia 1.7`.

1. `syncfiles.sh` syncs the file needed on Snellius to run the simulation.
2. `simulate.job` runs the simulation on Snellius using the `.env` variables.
3. `switchversions.jl` allows to switch between the local and snellius branches.

## Time estimation

For `N = 31` and `p = 11` processors the procedure takes approx. 13 minutes and 7 seconds. Worst scaling is `O(N^2)` and the target is `N = 201` and `p = 110`. So, the total time is estimated to be `13.07 * (11 / 110) * (201^2 / 31^2) = 55` minutes. Adding 20% for safety, the total time is estimated to be `1` hour and `6` minutes.
