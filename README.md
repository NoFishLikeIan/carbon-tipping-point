The following repository contains the code for the paper "The Cost of Uncertainty in Irreversible Climate Tipping Points" ([available here](https://www.andreatitton.com/static/docs/tipping-point.pdf)) and the working paper ``Climate Policy with Regional Tipping Points'' (forthcoming).

### Reproducing the paper

Running simulations (`scripts/runtipping.jl`, `scripts/rungame.jl` or `scripts/runbenchmark.jl`) requires `Julia 1.9.2`.

```bash
julia +1.9.2 --project scripts/runbenchmark.jl \
    -N 11 -s simulation/test-small \
    --overwrite -v 2 -p 8
```

All simulation scripts take the following arguments:

- `"--datapath"`, `"-d"`
    - type = `String`
    - default = "data"
    - help = "Path to data folder"

- `"--simulationpath"`, `"-s"`
    - type = `String`
    - default = "simulation/planner"
    - help = "Path to simulation folder"

- `"-N"`
    - type = `Int`
    - default = 51
    - help = "Size of grid"

- `"--verbose"` , "-v"
    - type = `Int`
    - default = 0
    - help = "Verbosity can be set to 0, 1, or 2 and larger."

- `"--overwrite"`

- `"--tol"`
    - type = `Float64`
    - default = 1e-5

- `"--stopat"`
    - type = `Float64`
    - default = 0.

- `"--cachestep"`
    - type = `Float64`
    - default = 0.25

### Snellius Supercomputer

The `jobs/` directory contains the scripts to run the simulations on the Snellius supercomputer.

### Input data  

Input data is available upon request. Contact me at a[dot]titton[at]uva[dot]nl.