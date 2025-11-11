using Pkg
Pkg.resolve(); Pkg.instantiate();

using Base.Threads: nthreads
using UnPack: @unpack
using Dates: now

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(ceargstable)

@unpack simulationdir, datapath, calibrationpath, dt = parsedargs
@unpack threshold, datapath, discovery = parsedargs
@unpack verbose = parsedargs

if (verbose ≥ 1)
    println("$(now()): ", "Running with $(nthreads()) threads with threshold $threshold and discovery $discovery...")
    flush(stdout)
end

# Begin script
using Model, Grid
using Base.Threads
using SciMLBase
using Statistics
using StaticArrays, SparseArrays
using Interpolations, DataStructures

using LinearSolve, LinearAlgebra

using JLD2, UnPack
using Dates, Printf

include("../src/valuefunction.jl")
include("../src/extend/model.jl")
include("../src/extend/grid.jl")
include("../src/extend/valuefunction.jl")

include("utils/saving.jl")
include("utils/simulating.jl")
include("markov/utils.jl")
include("markov/chain.jl")
include("markov/certaintyequivalence.jl")

simulationpath = joinpath(datapath, simulationdir)
@assert ispath(simulationpath)

begin # Load climate claibration
    climatepath = joinpath(datapath, "calibration", "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration = climatefile
    close(climatefile)
end

begin # Load linear model
    linearsolpath = joinpath(simulationpath, "linear", "growth", "logseparable", "negative", "Linear_burke_RRA10,00.jld2")
    @assert ispath(linearsolpath) "The linear simulation path does not exist: $linearsolpath"

    linearsimulation = loadtotal(linearsolpath);
end

begin # Load true threshold model
    thresholdkey = replace("T$(Printf.format(Printf.Format("%.2f"), threshold))", "." => ",")
    thresholdsolfile = "$(thresholdkey)_burke_RRA10,00.jld2"
    thresholdsolpath = joinpath(simulationpath, "tipping", "growth", "logseparable", "negative", thresholdsolfile)

    @assert ispath(thresholdsolpath) "The specified simulation file does not exist: $thresholdsolpath"

    thresholdsimulation = loadtotal(thresholdsolpath);
end

begin # Solve backward with discovery policy
    values, model, G = discoveryvalues(discovery, thresholdsimulation, linearsimulation)
    _, αitp = buildinterpolations(values, G);

    τ = maximum(keys(values))
    valuefunction = copy(values[τ])

    staticbackward!(valuefunction, dt, αitp, model, G, calibration; t₀ = 0., verbose = verbose, alg = KLUFactorization())
end

begin # Save
    outpath = joinpath(datapath, "ce", simulationdir)
    if !ispath(outpath) mkpath(outpath) end
    thresholdkey = replace("T$(Printf.format(Printf.Format("%.2f"), threshold))", "." => ",")
    discoverykey = replace("D$(Printf.format(Printf.Format("%.2f"), discovery))", "." => ",")
    outfile = joinpath(outpath, "$(thresholdkey)_$(discoverykey).jld2")

    if (verbose ≥ 1) println("$(now()): ", "Saving in ", outfile); flush(stdout) end

    JLD2.@save outfile threshold discovery valuefunction
end