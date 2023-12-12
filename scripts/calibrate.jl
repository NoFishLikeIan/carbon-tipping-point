using Revise

using DotEnv, UnPack
using CSV, DataFrames, JLD2

using DifferentialEquations
using DiffEqParamEstim, Optimization, OptimizationOptimJL

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
T0 = first(ipcctime)
T = last(ipcctime)

Mᵇ = bauscenario[:, "CO2 concentration"]
Tᵇ = bauscenario[:, "Temperature"]
Eᵇ = bauscenario[:, "CO2 emissions"] * 1e-9 # in Gton

begin # Calibrate growth rate γᵇ
    growthdata = Array(log.(Mᵇ)')
    
    function Fbau!(du, u, p, t)
        du[1] = Model.γ(t, p, T0)
    end

    calibrationproblem = ODEProblem(Fbau!, [growthdata[1]], extrema(ipcctime))
    
    cost = build_loss_objective(
        calibrationproblem, Tsit5(), L2Loss(ipcctime, growthdata), 
        Optimization.AutoForwardDiff();
        maxiters = 10000, verbose = false
    )

    optprob = Optimization.OptimizationProblem(cost, zeros(3))
    γparameters = solve(optprob, BFGS())
end

calibration = Model.Calibration(
    bauscenario.Year, 
    Eᵇ, 
    Tuple(γparameters)
)

save_object(joinpath(DATAPATH, "calibration.jld2"), calibration)