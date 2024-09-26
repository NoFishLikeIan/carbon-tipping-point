# Code for paper on "Regret and Tipping Points"

## Running simulations

Running simulations requires `Julia 1.9.2`. The simulations are run using the `scripts/runtipping.jl` or `scripts/runbenchmark.jl` script as follows 

```bash
julia +1.9.2 --project scripts/runbenchmark.jl -N 11 -s simulation/test-small --overwrite -v 2 -p 8
```

It takes the following command line arguments

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