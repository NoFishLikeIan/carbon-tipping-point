using Pkg
Pkg.resolve(); Pkg.instantiate();

using Base.Threads: nthreads
using UnPack: @unpack
using Dates: now

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(argtable)

@unpack datapath, simulationpath, overwrite = parsedargs # File system parameters
@unpack cachestep, verbose, stopat = parsedargs # IO parameters
@unpack NT, Nm, tol, dt, tau = parsedargs # Simulation parameters
@unpack threshold, damages, eis, rra, withnegative = parsedargs # Problem parameters

if !(eis ≈ 1)
    throw("Case ψ ≠ 1 not implemented yet!")
end

if (verbose ≥ 1)
    println("$(now()): ", "Running with $(nthreads()) threads...")

    if overwrite
        println("$(now()): ", "Running in overwrite mode!")
    end
    flush(stdout)
end

# Begin script
using Model, Grid
using Base.Threads
using SciMLBase
using Statistics
using StaticArrays, SparseArrays
using LinearSolve, LinearAlgebra

using Optimization, OptimizationOptimJL, LineSearches
using ForwardDiff

using JLD2
using Printf, Dates

include("../src/valuefunction.jl")
include("../src/extend/model.jl")
include("../src/extend/grid.jl")
include("../src/extend/valuefunction.jl")

include("utils/saving.jl")
include("markov/utils.jl")
include("markov/chain.jl")
include("markov/finitedifference.jl")

begin # Construct model
    DATAPATH = "data"
    calibrationpath = joinpath(DATAPATH, "calibration")

    # Load economic calibration
    abatementpath = joinpath(calibrationpath, "abatement.jld2")
    @assert isfile(abatementpath) "Abatement calibration file not found at $abatementpath"
    abatementfile = jldopen(abatementpath, "r+")
    @unpack abatement = abatementfile
    close(abatementfile)

    # Load climate claibration
    climatepath = joinpath(calibrationpath, "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = climatefile
    close(climatefile)

    damage = if damages == "kalkuhl"
        Kalkuhl{Float64}()
    elseif damages == "burke"
        BurkeHsiangMiguel{Float64}()
    elseif damages == "nodamages"
        NoDamageGrowth{Float64}()
    elseif damages == "weitzman"
        WeitzmanGrowth{Float64}()
    else
        error("Unknown damage type: $damages")
    end

    investments = Investment{Float64}()
    economy = Economy(investments = investments, damages = damage, abatement = abatement)
    
    decay = ConstantDecay(0.)
    climate = if 0 < threshold < Inf
        feedback = Model.updateTᶜ(threshold, feedback)
        TippingClimate(hogg, decay, feedback)
    else
        LinearClimate(hogg, decay)
    end

    preferences = LogSeparable(θ = rra);
    model = IAM(climate, economy, preferences)
end

begin # Construct Grid
    Tdomain = (0., 10.)  # Smaller, safer domain
    mmin = mstable(Tdomain[1] + 0.5, model.climate)
    mmax = mstable(Tdomain[2] - 0.5, model.climate)
    mdomain = (mmin, mmax)
    domains = (Tdomain, mdomain)
    N = (NT, Nm)
    Gterminal = RegularGrid(N, (Tdomain, mdomain))
end

if (verbose ≥ 1)
    modelstring = climate isa TippingClimate ? "tipping model with Tᶜ = $threshold," : "linear model with"

    println("$(now()): ","Solving $modelstring ψ = $eis, θ = $rra, $(withnegative ? "with" : "without") negative emissions and $damages damages...")
    flush(stdout)
end

outdir = joinpath(datapath, simulationpath)

if (verbose ≥ 1)
    println("$(now()): ","Running terminal...")
    flush(stdout)
end

tolerance = Error(tol, 1e-4)
terminalvaluefunction = ValueFunction(tau, climate, Gterminal, calibration)

Δt̄ = 1 / 12 # Steady state convergence is time step independent
equilibriumsteadystate!(terminalvaluefunction, Δt̄, linearIAM(model), Gterminal, calibration; timeiterations = 200_000, verbose, tolerance)
steadystate!(terminalvaluefunction, Δt̄, model, Gterminal, calibration; timeiterations = 200_000, verbose, tolerance, withnegative)

if (verbose ≥ 1)
    println("$(now()): ","Running backward...")
    flush(stdout)
end

G = shrink(Gterminal, (0.05, 0.05))
valuefunction = interpolateovergrid(terminalvaluefunction, Gterminal, G)
backwardsimulation!(valuefunction, dt, model, G, calibration; verbose, withnegative, overwrite, outdir, cachestep = cachestep, startcache = 150.)