using ArgParse
using UnPack: @unpack
using Dates: now
using Base.Threads: nthreads

using Model, Grid
using SciMLBase
using Statistics
using StaticArrays, SparseArrays
using Interpolations, FastChebInterp, DataStructures
using FastClosures

using Optimization, OptimizationOptimJL, OptimizationBase
using LinearSolve, LinearAlgebra

using JLD2
using Printf, Dates

include("../src/valuefunction.jl")
include("../src/extend/model.jl")
include("../src/extend/grid.jl")
include("../src/extend/valuefunction.jl")
include("../src/regret.jl")

include("utils/saving.jl")
include("utils/simulating.jl")
include("utils/loading.jl")
include("utils/approximate.jl")
include("markov/chain.jl")
include("markov/finitedifference.jl")
include("regret/chain.jl")
include("regret/finitedifference.jl")

function optimize_weights(; datapath = "data",
                            simulationpath = "data/simulation",
                            verbose = 0,
                            dt = 1 / 12,
                            tau = 150.0,
                            damages = "burke",
                            K = 10,
                            coarsen_x = 4,
                            coarsen_m = 4,
                            cheborder_T = 20,
                            cheborder_m = 20,
                            cheborder_t = 10,
                            cheborder_c = 5,
                            optimizer = "cobyla",
                            maxiter = 1000,
                            tol = 1e-2,
                            steadystatetolerance = 1e-3)

    if verbose ≥ 1
        println("$(now()): Running min-max regret optimization with $(nthreads()) threads...")
        flush(stdout)
    end

    calibrationpath = joinpath(datapath, "calibration")

    # Load economic calibration
    abatementpath = joinpath(calibrationpath, "abatement.jld2")
    @assert isfile(abatementpath) "Abatement calibration file not found at $abatementpath"
    abatementfile = jldopen(abatementpath, "r+")
    @unpack abatement = abatementfile
    close(abatementfile)

    investments = Investment()
    damages_obj = if damages == "burke"
        BurkeHsiangMiguel()
    elseif damages == "kalkuhl"
        Kalkuhl()
    elseif damages == "weitzman"
        WeitzmanGrowth()
    else
        error("Unknown damage type: $damages")
    end
    economy = Economy(investments = investments, damages = damages_obj, abatement = abatement)

    # Load climate calibration
    climatepath = joinpath(calibrationpath, "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration, hogg, feedback = climatefile
    close(climatefile)

    # Initialize model
    Δt = dt
    τ = tau
    preferences = LogSeparable()
    decay = ConstantDecay(0.)
    climate = LinearClimate(hogg, decay)
    linearmodel = IAM(climate, economy, preferences)

    if verbose ≥ 1
        println("$(now()): Loading simulation paths...")
        flush(stdout)
    end

    # Construct policy simplex
    paths = loadsimulationpaths(simulationpath; exclude = ["terminal", "linear"])
    _, G = loadproblem(paths[2.0])

    filteredpaths = basispolicypaths(paths, K, G; tspan = (0, 10))
    policybasis = SimplexPolicies(filteredpaths)

    if verbose ≥ 1
        println("$(now()): Building Chebyshev representation...")
        flush(stdout)
    end

    # Construct Chebyshev representation of full information
    values = OrderedDict(k => loadtotal(p) for (k, p) in paths)
    cheborder = (cheborder_T, cheborder_m, cheborder_t, cheborder_c)
    H̃ = chebyshevrepresentation(values, cheborder)

    # Setup optimization problem
    if verbose ≥ 1
        println("$(now()): Setting up optimization problem...")
        flush(stdout)
    end

    G = coarse(G, (coarsen_x, coarsen_m))
    valuefunction = ValueFunction(τ, climate, G, calibration)

    weights = MVector{K}(rand(K))
    weights ./= sum(weights)

    # Define regret function
    function regret_closure(weights, threshold)
        valuefunction.t.t = τ
        
        climate_tipping = TippingClimate(linearmodel.climate.hogg, linearmodel.climate.decay, updatethreshold(threshold, feedback))
        model = IAM(climate_tipping, linearmodel.economy, linearmodel.preferences)

        tol_error = Error{eltype(G)}(steadystatetolerance, steadystatetolerance)
        steadystate!(valuefunction, weights, Δt, model, G, calibration, policybasis; tolerance = tol_error, verbose = verbose - 1)
        backwardsimulation!(valuefunction, weights, Δt, model, G, calibration, policybasis; verbose = verbose - 1)

        x₀ = Point(climate_tipping.hogg.T₀, log(climate_tipping.hogg.M₀ / climate_tipping.hogg.Mᵖ))
        Gⱼ = interpolateovergrid(valuefunction.H, G, x₀)
        Hⱼ = H̃(SVector(x₀.T, x₀.m, zero(τ), threshold))

        return Gⱼ - Hⱼ
    end

    # Define max regret objective
    function maxregret_closure(weights, optparameters)
        maxregretobj = @closure threshold -> regret_closure(weights, threshold)
        r, _ = gss(maxregretobj, H̃.lb[4], H̃.ub[4]; tol = tol)
        return r
    end

    if verbose ≥ 1
        println("$(now()): Starting optimization...")
        flush(stdout)
    end

    minmaxregretobj = OptimizationFunction(maxregret_closure)
    optparameters = nothing
    minmaxregretprob = OptimizationProblem(minmaxregretobj, weights, optparameters)

    opt_alg = if optimizer == "cobyla"
        COBYLA(; rhoend = tol)
    elseif optimizer == "neldermead"
        NelderMead()
    else
        error("Unknown optimizer: $optimizer")
    end

    sol = solve(minmaxregretprob, opt_alg; maxiters = maxiter)

    if verbose ≥ 1
        println("$(now()): Optimization complete")
        println("  Optimal weights: $(sol.u)")
        println("  Min-max regret: $(sol.minimum)")
        flush(stdout)
    end

    return sol
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--datapath"
            help = "Path to data directory"
            arg_type = String
            default = "data"
        "--simulationpath"
            help = "Path to simulation data"
            arg_type = String
            default = "data/simulation"
        "--verbose", "-v"
            help = "Verbosity level"
            arg_type = Int
            default = 1
        "--dt"
            help = "Time step"
            arg_type = Float64
            default = 1.0 / 12.0
        "--tau"
            help = "Terminal time"
            arg_type = Float64
            default = 150.0
        "--damages"
            help = "Damage specification: burke, kalkuhl, weitzman"
            arg_type = String
            default = "burke"
        "--K"
            help = "Number of policy basis functions"
            arg_type = Int
            default = 10
        "--coarsen-x"
            help = "Grid coarsening factor (T dimension)"
            arg_type = Int
            default = 4
        "--coarsen-m"
            help = "Grid coarsening factor (m dimension)"
            arg_type = Int
            default = 4
        "--optimizer"
            help = "Optimizer: cobyla, neldermead"
            arg_type = String
            default = "cobyla"
        "--maxiter"
            help = "Maximum iterations for optimizer"
            arg_type = Int
            default = 1000
        "--tol"
            help = "Optimization tolerance"
            arg_type = Float64
            default = 1e-2
    end

    return parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    sol = optimize_weights(; 
                           datapath = args["datapath"],
                           simulationpath = args["simulationpath"],
                           verbose = args["verbose"],
                           dt = args["dt"],
                           tau = args["tau"],
                           damages = args["damages"],
                           K = args["K"],
                           coarsen_x = args["coarsen-x"],
                           coarsen_m = args["coarsen-m"],
                           optimizer = args["optimizer"],
                           maxiter = args["maxiter"],
                           tol = args["tol"])
end

function maxregret(parsedargs::AbstractDict)
    kwargs = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in pairs(parsedargs))
    return maxregret(; kwargs...)
end

function maxregret(parsedargs::NamedTuple)
    return maxregret(; parsedargs...)
end

function maxregret(; datapath = "data",
                    simulationpath = "data/simulation",
                    overwrite = false,
                    verbose = 0,
                    dt = 1 / 12,
                    tau = 150.0,
                    damages = "burke",
                    rra = 10.0,
                    K = 10,
                    coarsen = (4, 4),
                    cheborder = (20, 20, 10, 5),
                    optimizer = "cobyla",
                    maxiter = 1000,
                    tol = 1e-2,
                    steadystatetolerance = 1e-3,
                    Tdomain = (0., 10.)
    )

    if verbose ≥ 1
        println("$(now()): Running min-max regret optimization with $(nthreads()) threads...")
        flush(stdout)
    end

    calibrationpath = joinpath(datapath, "calibration")

    # Load economic calibration
    abatementpath = joinpath(calibrationpath, "abatement.jld2")
    @assert isfile(abatementpath) "Abatement calibration file not found at $abatementpath"
    abatementfile = jldopen(abatementpath, "r+")
    @unpack abatement = abatementfile
    close(abatementfile)

    investments = Investment()
    damages_obj = if damages == "burke"
        BurkeHsiangMiguel()
    elseif damages == "kalkuhl"
        Kalkuhl()
    elseif damages == "weitzman"
        WeitzmanGrowth()
    else
        error("Unknown damage type: $damages")
    end
    economy = Economy(investments = investments, damages = damages_obj, abatement = abatement)

    # Load climate calibration
    climatepath = joinpath(calibrationpath, "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration, hogg, feedback = climatefile
    close(climatefile)

    # Initialize model
    Δt = dt
    τ = tau
    preferences = LogSeparable()
    decay = ConstantDecay(0.)
    climate = LinearClimate(hogg, decay)
    linearmodel = IAM(climate, economy, preferences)

    if verbose ≥ 1
        println("$(now()): Loading simulation paths...")
        flush(stdout)
    end

    # Construct policy simplex
    paths = loadsimulationpaths(simulationpath; exclude = ["terminal", "linear"])
    _, G = loadproblem(paths[2.0])

    filteredpaths = basispolicypaths(paths, K, G; tspan = (0, 10))
    policybasis = SimplexPolicies(filteredpaths)

    if verbose ≥ 1
        println("$(now()): Building Chebyshev representation...")
        flush(stdout)
    end

    # Construct Chebyshev representation of full information
    values = OrderedDict(k => loadtotal(p) for (k, p) in paths)
    H̃ = chebyshevrepresentation(values, cheborder)

    # Setup optimization problem
    if verbose ≥ 1
        println("$(now()): Setting up optimization problem...")
        flush(stdout)
    end

    G = coarse(G, coarsen)
    valuefunction = ValueFunction(τ, climate, G, calibration)

    weights = MVector{K}(rand(K))
    weights ./= sum(weights)

    function regret(weights, threshold, H̃, valuefunction, τ, Δt, (linearmodel, feedback), G, calibration, policybasis; steadystatetolerance = Error{eltype(G)}(steadystatetolerance, steadystatetolerance), verbose = 0)
        valuefunction.t.t = τ
        
        climate = TippingClimate(linearmodel.climate.hogg, linearmodel.climate.decay, updatethreshold(threshold, feedback))
        model = IAM(climate, linearmodel.economy, linearmodel.preferences)

        steadystate!(valuefunction, weights, Δt, model, G, calibration, policybasis; tolerance = steadystatetolerance, verbose = verbose)
        backwardsimulation!(valuefunction, weights, Δt, model, G, calibration, policybasis; verbose = verbose)

        x₀ = Point(climate.hogg.T₀, log(climate.hogg.M₀ / climate.hogg.Mᵖ))
        Gⱼ = interpolateovergrid(valuefunction.H, G, x₀)
        Hⱼ = H̃(SVector(x₀.T, x₀.m, zero(τ), threshold))

        return Gⱼ - Hⱼ
    end

    function maxregretobj(weights, optparameters)
        H̃, valuefunction, τ, Δt, (linearmodel, feedback), G, calibration, policybasis = optparameters
        
        maxregretobj = @closure threshold -> regret(weights, threshold, H̃, valuefunction, τ, Δt, (linearmodel, feedback), G, calibration, policybasis; verbose = verbose - 1)
        
        r, _ = gss(maxregretobj, H̃.lb[4], H̃.ub[4]; tol = tol)
        
        return r
    end

    optparameters = H̃, valuefunction, τ, Δt, (linearmodel, feedback), G, calibration, policybasis

    if verbose ≥ 1
        println("$(now()): Starting optimization...")
        flush(stdout)
    end

    minmaxregretobj = OptimizationFunction(maxregretobj)
    minmaxregretprob = OptimizationProblem(minmaxregretobj, weights, optparameters)

    opt_alg = if optimizer == "cobyla"
        COBYLA(; rhoend = tol)
    elseif optimizer == "neldermead"
        NelderMead()
    else
        error("Unknown optimizer: $optimizer")
    end

    sol = solve(minmaxregretprob, opt_alg; maxiters = maxiter)

    if verbose ≥ 1
        println("$(now()): Optimization complete")
        println("  Optimal weights: $(sol.u)")
        println("  Min-max regret: $(sol.minimum)")
        flush(stdout)
    end

    return sol
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--datapath"
            help = "Path to data directory"
            arg_type = String
            default = "data"
        "--simulationpath"
            help = "Path to simulation data"
            arg_type = String
            default = "data/simulation"
        "--overwrite"
            help = "Overwrite existing results"
            action = :store_true
        "--verbose", "-v"
            help = "Verbosity level"
            arg_type = Int
            default = 1
        "--dt"
            help = "Time step"
            arg_type = Float64
            default = 1.0 / 12.0
        "--tau"
            help = "Terminal time"
            arg_type = Float64
            default = 150.0
        "--damages"
            help = "Damage specification: burke, kalkuhl, weitzman"
            arg_type = String
            default = "burke"
        "--rra"
            help = "Relative risk aversion"
            arg_type = Float64
            default = 10.0
        "--K"
            help = "Number of policy basis functions"
            arg_type = Int
            default = 10
        "--coarsen"
            help = "Grid coarsening factor"
            arg_type = Tuple{Int, Int}
            default = (4, 4)
        "--optimizer"
            help = "Optimizer: cobyla, neldermead"
            arg_type = String
            default = "cobyla"
        "--maxiter"
            help = "Maximum iterations for optimizer"
            arg_type = Int
            default = 1000
        "--tol"
            help = "Optimization tolerance"
            arg_type = Float64
            default = 1e-2
    end

    return parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    sol = maxregret(args)
end
