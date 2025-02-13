using Revise

using DotEnv, UnPack
using CSV, DataFrames, JLD2

using DifferentialEquations
using DiffEqParamEstim, Optimization, OptimizationOptimJL
using Roots, FastClosures

using Plots

using Model

env = DotEnv.config()

BASELINE_YEAR = parse(Int64, get(env, "BASELINE_YEAR", "2020"))
DATAPATH = get(env, "DATAPATH", "data/")
SPPNAME = get(env, "SSPNAME", "")
IPCCDATAPATH = joinpath(DATAPATH, SPPNAME, "$SPPNAME.csv")
MEDIANPATH = joinpath(DATAPATH, "climate-data", "proj-median.csv")

@assert isfile(IPCCDATAPATH)
@assert isfile(MEDIANPATH)

# Import IPCC data
medianraw = CSV.read(MEDIANPATH, DataFrame)
rawdata = CSV.read(IPCCDATAPATH, DataFrame); select!(rawdata, Not(:MODEL));

filters = (
    :SCENARIO => ByRow(==("SSP5-85 (Baseline)")),
    :UNIT => ByRow(==("Mt CO2/yr")),
    :VARIABLE => ByRow(==("CMIP6 Emissions|CO2"))
)

begin
    baudata = subset(rawdata, filters...); 
    select!(baudata, Not(first.(filters)...));
    baudata = permutedims(baudata, 1);
    rename!(baudata, :REGION => :YEAR);
    baudata[!, :YEAR] .= parse.(Int, baudata[!, :YEAR]);

    baudata[!, Not(:YEAR)] .*= 0.001 # Mt to Gt
end

# World average calibration
Eᵇ = baudata.World
time = baudata.YEAR .- BASELINE_YEAR

T0 = Float64(first(time))
T = Float64(last(time))

Mᵇ = filter(:Scenario => isequal("SSP5 - Baseline"), medianraw)[2:end, "CO2 concentration"]

function parametricemissions(u, p, t)
    Δt = t - T0
    p[1] + p[2] * Δt + p[3] * Δt^2
end

function extractparameters(growthdata; p₀ = 0.2 * zeros(3), maxiters = 10_000, verbose = false)
    m₀ = first(growthdata)

    calibrationproblem = ODEProblem{false}(parametricemissions, m₀, extrema(time); p = p₀)

    cost = build_loss_objective(calibrationproblem, Tsit5(), L2Loss(time, growthdata), Optimization.AutoForwardDiff(); maxiters, verbose, saveat = time)

    optprob = Optimization.OptimizationProblem(cost, p₀)
    γparameters = solve(optprob, BFGS())

    p₀, p₁, p₂ = γparameters.u
    Δ = (T - T0)
    r = -(p₁ + 2p₂ * Δ) / (p₀ + p₁ * Δ + p₂ * Δ^2)

    return Tuple(γparameters), r
end

begin # Calibrate growth rate γᵇ
    growthdata = log.(Mᵇ)

    γparameters, r = extractparameters(growthdata)
end

# Plot solution
let
    solvedprob = ODEProblem{false}(parametricemissions, growthdata[1], extrema(time); p = γparameters)
    sol = solve(solvedprob)

    plot(time, growthdata; label = "IPCC", marker = :o)
    plot!(time, t -> sol(t); label = "Solved", marker = :o)
end

calibration = Calibration(time, Eᵇ, γparameters, r, (T0, T))
save_object(joinpath(DATAPATH, "calibration.jld2"), calibration)

# Regional calibration
oecdfrac = baudata.var"R5.2OECD" ./ Eᵇ
regionalcalibration = RegionalCalibration(calibration, oecdfrac)

save_object(joinpath(DATAPATH, "regionalcalibration.jld2"), regionalcalibration)

let
    fig = plot(xlabel = "Year")
    ts = range(0., 80.; length = 101)

    growthrate = γ.(ts, regionalcalibration)

    plot!(fig, ts, first.(growthrate); label = "OECD")
    plot!(fig, ts, last.(growthrate); label = "RoW")
    plot!(fig, ts, sum.(growthrate); label = "World")

    fig
end

# Climate sensitivity
thresholds = [1.5, 2.5];
const hogg = Hogg();

function deviation(Δλ, Tᶜ)
    maximum(find_zeros(T -> mstable(T, hogg, Albedo(Tᶜ, 1.8, 0.31, Δλ)) - log(2hogg.Mᵖ), hogg.Tᵖ .+ (0., 12.))) - hogg.Tᵖ
end

sols = Albedo[]
for Tᶜ in thresholds
    Δλ = find_zero(Δλ -> deviation(Δλ, Tᶜ) - 4.5, [0., 0.1])
    albedo = Albedo(Tᶜ, 1.8, 0.31, Δλ)
end