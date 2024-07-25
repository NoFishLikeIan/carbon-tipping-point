using Revise

using DotEnv, UnPack
using CSV, DataFrames, JLD2

using DifferentialEquations
using DiffEqParamEstim, Optimization, OptimizationOptimJL

using Plots

using Model

env = DotEnv.config()

BASELINE_YEAR = parse(Int64, get(env, "BASELINE_YEAR", "2020"))
DATAPATH = get(env, "DATAPATH", "data/") 
IPCCDATAPATH = joinpath(DATAPATH, "climate-data", "proj-median.csv")

# Import IPCC data
ipccproj = CSV.read(IPCCDATAPATH, DataFrame)

getscenario(s::Int64) = filter(:Scenario => isequal("SSP$(s) - Baseline"), ipccproj)

bauscenario = getscenario(5)
ipcctime = bauscenario.Year .- BASELINE_YEAR
T0 = Float64(first(ipcctime))
T = Float64(last(ipcctime))

Mᵇ = bauscenario[:, "CO2 concentration"]
Tᵇ = bauscenario[:, "Temperature"]
Eᵇ = bauscenario[:, "CO2 emissions"] * 1e-9 # in Gton

begin # Calibrate growth rate γᵇ
    growthdata = Array(log.(Mᵇ)')
    
    function parametricemissions(u, p, t)
        Δt = t - T0
        p[1] + p[2] * Δt + p[3] * Δt^2
    end

    p₀ = rand(3)
    calibrationproblem = ODEProblem{false}(parametricemissions, growthdata[1], extrema(ipcctime); p = p₀)
    
    cost = build_loss_objective(
        calibrationproblem, Tsit5(), L2Loss(ipcctime, growthdata), 
        Optimization.AutoForwardDiff();
        maxiters = 10_000, verbose = false, saveat = ipcctime
    )

    optprob = Optimization.OptimizationProblem(cost, p₀)
    γparameters = solve(optprob, BFGS())
end

# Plot solution
begin
    solvedprob = ODEProblem{false}(parametricemissions, growthdata[1], extrema(ipcctime); p = γparameters)
    sol = solve(solvedprob)

    plot(ipcctime, growthdata'; label = "IPCC", marker = :o)
    plot!(ipcctime, t -> sol(t); label = "Solved", marker = :o)
end

calibration = Model.Calibration(bauscenario.Year, Eᵇ, Tuple(γparameters), (T0, T))
save_object(joinpath(DATAPATH, "calibration.jld2"), calibration)