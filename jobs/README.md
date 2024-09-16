# Jobs

Some bash and julia scripts to speed up multiple versioning. This is needed to run scripts on Snellius since I locally use `julia 1.10` and remotely `julia 1.7`.

1. `syncfiles.sh` syncs the file needed on Snellius to run the simulation.
2. `simulate.job` runs the simulation on Snellius using the `.env` variables.
3. `switchversions.jl` allows to switch between the local and snellius branches.

## Time estimation

For `N = 11` the procedure takes approx. 5 minutes and 6 seconds. Worst scaling is `O(N^2)` and the target is `N = 201`. So, the total time is estimated to be `5.1 * 201^2 = 205,000` seconds or `57` hours. Adding 20% for safety, the total time is estimated to be `68` hours.
Some bash and julia scripts to speed up multiple versioning. This is needed to run scripts on Snellius since I locally use `julia 1.10` and remotely `julia 1.7`.
