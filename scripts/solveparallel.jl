using Distributed
using Dates: now

## Inputs
const workers = 8
const datapath = "./data"
const simulationpath = "simulation-test"

addprocs(workers - 1, exeflags = "--project=$(Base.active_project())")

## Definition of parameters
const thresholds = [2:0.05:4...,-1.0]
const damages = ["burke"]
const withnegatives = [true]

const overwrite = true
const cachestep = 0.5
const verbose = 0
const stopat = 0.0
const nt = 200
const nm = 250
const tol = 1e-3
const dt = 0.005
const tau = 500.0
const eis = 1.0
const rra = 10.0
const tdomain = (0.0, 10.0)

const kwargslist = NamedTuple[]
for damage in damages, withnegative in withnegatives, threshold in thresholds
	push!(kwargslist, (
		datapath = datapath,
		simulationpath = simulationpath,
		overwrite = overwrite,
		cachestep = cachestep,
		verbose = verbose,
		stopat = stopat,
		NT = nt,
		Nm = nm,
		tol = tol,
		dt = dt,
		tau = tau,
		threshold = threshold,
		damages = damage,
		eis = eis,
		rra = rra,
		withnegative = withnegative,
		Tdomain = tdomain
	))
end

## Run script
@everywhere begin
    using Pkg
    Pkg.instantiate(); Pkg.precompile()
    include("solve.jl")
end

if verbose ≥ 1
	println("$(now()): Running $(length(kwargslist)) parameter sets across $(nprocs()) processes...")
end

const results = pmap(kwargslist) do kwargs
	try
		solve(; kwargs...)
		(ok = true, threshold = kwargs.threshold, damages = kwargs.damages, withnegative = kwargs.withnegative)
	catch err
		(ok = false, threshold = kwargs.threshold, damages = kwargs.damages, withnegative = kwargs.withnegative, error = sprint(showerror, err))
	end
end

const failures = filter(r -> !r.ok, results)
if !isempty(failures)
	println("$(now()): $(length(failures)) run(s) failed.")
	for failure in failures
		println("  threshold=$(failure.threshold), damages=$(failure.damages), withnegative=$(failure.withnegative): $(failure.error)")
	end
elseif verbose ≥ 1
	println("$(now()): All runs completed successfully.")
end
