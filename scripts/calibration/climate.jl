using Revise

using DotEnv, UnPack, DataStructures
using CSV, DataFrames, JLD2
using StaticArrays
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis
using Optimization, OptimizationOptimJL, OptimizationPolyalgorithms
using ForwardDiff, DifferentiationInterface

using Roots, FastClosures
using Statistics, LinearAlgebra
using Interpolations
using LogExpFunctions

using Plots, Printf, PGFPlotsX, Colors, ColorSchemes, LaTeXStrings

# pgfplotsx()
default(label = false, dpi = 180, linewidth = 2.5)
push!(PGFPlotsX.CUSTOM_PREAMBLE,
    raw"\usepgfplotslibrary{fillbetween}",
    raw"\usepackage{siunitx}",
    raw"\DeclareSIUnit{\ppm}{p.p.m.}"
);

using Model, Grid

includet("../../src/valuefunction.jl")
includet("../../src/extend/model.jl")
includet("../utils/saving.jl")
includet("../utils/simulating.jl")
includet("constants.jl")

PLOTPATH = "papers/job-market-paper/submission/plots"
DATAPATH = "data"
SAVEFIG = false
PALETTE = colorschemes[:grays];
calibrationpath = joinpath(DATAPATH, "calibration")
if !isdir(calibrationpath) mkpath(calibrationpath) end

# Loading data 
function parsescenario(sspkey)
    m = match(r"ssp(\d)(\d{2})", sspkey)
    ssp, scenario = m.captures

    return String15("SSP$ssp $(scenario[1]).$(scenario[2])")
end
function safeparse(Type, value)
    if ismissing(value)
        return NaN
    elseif value isa Type
        return value
    else
        return parse(Type, value)
    end
end

function loadhierdataframe(variablepathpair::Pair{K,String}; groupkeys=[:Group]) where K<:Union{String,Symbol}
    ngroups = length(groupkeys)
    variablename, filepath = variablepathpair

    df = DataFrame(CSV.File(filepath))

    years = parse.(Int, df[(2+ngroups):end, 1])
    nyears = length(years)

    scenarios = map(parsescenario, names(df)[2:end])
    nscenarios = length(scenarios)

    variables = [
        groupkey => repeat(collect(df[i, 2:end]), inner=nyears)
        for (i, groupkey) in enumerate(groupkeys)
    ]

    data = safeparse.(Float64, df[(2+ngroups):end, 2:end] |> Matrix)
    vecdata = vec(data)

    newdf = DataFrame(
        :Year => repeat(years, outer=nscenarios),
        :Scenario => repeat(scenarios, inner=nyears),
        variables...,
        Symbol(variablename) => safeparse.(Float64, vecdata)
    )

    return newdf
end

coupleddir = joinpath(DATAPATH, "deutloff", "model_output", "coupled_ensemble");
@assert isdir(coupleddir);
uncoupleddir = joinpath(DATAPATH, "deutloff", "model_output", "uncoupled_ensemble");
@assert isdir(uncoupleddir);

begin # Load temperature dataframe
    temperaturedf = loadhierdataframe(:T_Coupled => joinpath(coupleddir, "T.csv"); groupkeys=[:Quantile])
    temperaturedf.T_Uncoupled = loadhierdataframe(:T_Uncoupled => joinpath(uncoupleddir, "T.csv"); groupkeys=[:Quantile]).T_Uncoupled

    temperature = groupby(temperaturedf, [:Scenario, :Quantile])
end;

begin # Load emissions and GHGs dataframes
    concentrationdf = loadhierdataframe("Concentration" => joinpath(coupleddir, "C.csv"); groupkeys=[:Particle, :Quantile])
    concentrationdf.Quantile = safeparse.(Float64, concentrationdf.Quantile)

    emissionsdf = loadhierdataframe("Emissions" => joinpath(coupleddir, "Emm.csv"); groupkeys=[:Particle, :Quantile])
    emissionsdf.Quantile = safeparse.(Float64, emissionsdf.Quantile)

    concentrationdf.Emissions = emissionsdf.Emissions
    concentration = groupby(concentrationdf, [:Scenario, :Particle, :Quantile])
end;

if isinteractive() # Figure CO₂ concentration
    qs = SVector(0.05, 0.5, 0.95)
    figscenarios = SVector("SSP1 1.9", "SSP2 4.5", "SSP4 3.4", "SSP5 8.5")
    cmap = palette(:viridis, length(figscenarios); rev=true)

    co2fig = plot(xlabel="Year", yaxis="Concentration (ppm)", title="Carbon Dioxide Concentration", xlims=(2000, 2400))
    excesstfig = plot(xlabel="Year", yaxis="Temperature [°]", title="Additional temperature", xlims=(2012, 2400))

    for (i, scenario) in enumerate(figscenarios)
        co2 = Tuple(concentration[(scenario, "carbon_dioxide", q)] for q in qs)
        temp = Tuple(temperature[(scenario, q)] for q in qs)

        c = cmap[i]

        plot!(co2fig, co2[1].Year, co2[1].Concentration; color=c, fillrange=co2[3].Concentration, fillalpha=0.05, linewidth=0., label=false)
        plot!(co2fig, co2[2].Year, co2[2].Concentration; label=scenario, color=c, linewidth=2.5)

        plot!(excesstfig, temp[1].Year, temp[1].T_Coupled - temp[1].T_Uncoupled; color=c, fillrange=temp[3].T_Coupled - temp[3].T_Uncoupled, fillalpha=0.05, linewidth=0., label=false)
        plot!(excesstfig, temp[2].Year, temp[2].T_Coupled - temp[2].T_Uncoupled; label=scenario, color=c, linewidth=2.5)
    end

    impulsefig = plot(co2fig, excesstfig; layout=(1, 2), size=600 .* (2√2, 1), margins=5Plots.mm, legend=:topleft)
end

"Construct CO2e concentration"
function computeco2equivalence(concentration, q, gwpdict)
    co2equivalence = deepcopy(concentration[("SSP5 8.5", "carbon_dioxide", q)])
    co2equivalence[!, "CO2 Concentration"] .= co2equivalence[:, "Concentration"]
    co2equivalence[!, "CO2 Emissions"] .= co2equivalence[:, "Emissions"]

    # Concentration
    co2econcentration = copy(co2equivalence[:, "Concentration"])
    for (particle, (gwpvalue, concentrationconverter, factor)) in gwpdict
        df = concentration[("SSP5 8.5", particle, q)]
        
        # Convert to ppm, then weight by GWP for radiative equivalence
        concppm = df.Concentration .* concentrationconverter * factor
        co2econcentration .+= concppm .* gwpvalue
        co2equivalence[!, "$particle Concentration"] .= concppm
        co2equivalence[!, "$particle CO2e Concentration"] .= concppm .* gwpvalue
    end
    co2equivalence[!, "Concentration"] .= co2econcentration

    # Emissions
    co2masstoconcentration = gwpdict["carbon_dioxide"][3]
    co2eemissions = co2equivalence[:, "CO2 Emissions"] .* co2masstoconcentration
    
    # Add other gases: each contributes ppm/yr based on its own molecular weight and GWP
    for (particle, (gwpvalue, concentrationconverter, factor)) in gwpdict
        df = concentration[("SSP5 8.5", particle, q)]
        emissionsppmfactor = concentrationconverter * factor

        emissions = df.Emissions .* emissionsppmfactor
        co2eemissions .+= emissions .* gwpvalue
        co2equivalence[!, "$particle Emissions"] .= emissions
    end
    co2equivalence[!, "Emissions"] .= co2eemissions

    return co2equivalence
end

begin
    println("Molecular weights (g/mol):")
    for (gas, mw) in molweights
        println("  $gas: $mw")
    end
    println("\nMass-to-concentration factors (Gt → ppm):")
    for (gas, factor) in masstoconcentration
        println("  $gas: $(round(factor, digits=4))")
    end
    println("\nGWP values (AR6, 100-year):")
    for (gas, gwpval) in gwpvalues
        println("  $gas: $(gwpval)")
    end

    # GWP dictionary: (gwp_value, concentration_unit_factor, emission_unit_factor, mass_to_conc_factor)
    gwpdict = Dict(mol => (gwpvalue, converter[mol], masstoconcentration[mol]) for (mol, gwpvalue) in gwpvalues)

    co2equivalence = computeco2equivalence(concentration, 0.5, gwpdict)
    co2equivalencelower = computeco2equivalence(concentration, 0.05, gwpdict)
    co2equivalenceupper = computeco2equivalence(concentration, 0.95, gwpdict)
end;

# Define the no-policy scenario for calibration
npscenario = "SSP5 8.5"

if isinteractive() # Figure CO₂ concentration in the no policy scenario
    co2equivalencefig = plot(xlabel="Year", yaxis="Concentration (ppm)", title="Carbon Dioxide Concentration in SSP5 8.5", xlims=(1900, 2200), legend = :topleft)

    concentrationnames = filter(m -> occursin("Concentration", m), names(co2equivalence))
    colors = palette(:tab10, length(concentrationnames); rev=true)

    for (i, cname) in enumerate(concentrationnames)
        color = colors[i]
        plot!(co2equivalencefig, co2equivalence.Year, co2equivalence[:, cname]; label=cname, linewidth=2.5, c=color)
        plot!(co2equivalencefig, co2equivalencelower.Year, co2equivalencelower[:, cname]; label=false, fillrange=co2equivalenceupper[:, cname] .+ 1e-3, fillalpha=0.2, linewidth=0., c=color)
    end

    co2equivalencefig
end

if isinteractive() # Figure fraction of forcing
    yearbound = (1980, 2200)
    tdxs = @. yearbound[1] ≤ co2equivalence.Year ≤ yearbound[2]

    xtick = collect(range(yearbound...; step=20))
    xticklabelposition = range(yearbound...; step=40)
    xticklabels = [t ∈ xticklabelposition ? "\$$t\$" : "" for t in xtick]

    ytick = collect(0.7:0.05:1)
    yticklabels = [@sprintf "%.0f\\%%" 100t for t in ytick]

    fracradiation = co2equivalence[tdxs, "CO2 Concentration"] ./ co2equivalence[tdxs, "Concentration"]

    fracfig = @pgf Axis({
        width = raw"0.7\textwidth",
        height = raw"0.5\textwidth",
        grid = "both",
        xlabel = raw"Year, $t$",
        ylabel = raw"$\hat{M}^{\mathrm{np}}_{t, \mathrm{CO}_2} / \hat{M}^{\mathrm{np}}_t$",
        title = raw"Fraction of $\mathrm{CO}_2$ to $\mathrm{CO}_2$e concentration",
        ylabel_style = {align = "center"},
        xmin = yearbound[1], xmax = yearbound[2],
        xtick = xtick, xticklabels = xticklabels,
        ytick = ytick, yticklabels = yticklabels,
        ymin = 0.69, ymax = 1.0,
    })

    curve = @pgf Plot({ line_width = 2.5 }, Coordinates(co2equivalence.Year[tdxs], fracradiation))

    push!(fracfig, curve)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "fracfig.tikz"), fracfig; include_preamble=true)
    end

    fracfig
end

# --- Computing parametric emissions form
begin # Setup CO₂e maximisation problem
    baselineyear = 2020.
    τ = 2200. - baselineyear
    co2tspan = baselineyear .+ (0., τ)

    tdxs = baselineyear .≤ co2equivalence.Year .≤ (baselineyear + τ)
    co2calibrationdf = co2equivalence[tdxs, :]

    Mᵖ = mean(co2equivalence[1800 .≤ co2equivalence.Year .≤ 1900, "Concentration"])
    m = @. log(co2calibrationdf.Concentration / Mᵖ)
    t = co2calibrationdf.Year[1:end-1]
    γ̂ₜ = diff(m); smooth!(γ̂ₜ, 5)  # Reduced smoothing to preserve SSP5-8.5 growth dynamics
    Eₜ = Vector{Float64}(co2calibrationdf.Emissions)
end;

function growthrate(t, p)
    γ₀, α, γ₁, β, γ̄ = p

    return γ₀ * exp(α * t) + γ₁ * exp(β * t) + γ̄
end

begin
    growthloss = @closure p -> mean(abs2, growthrate.(t .- co2tspan[1], Ref(p)) .- γ̂ₜ)
    p₀ = MVector(0.001, -0.02, -0.01, -0.02, 0.002)

    γres = optimize(growthloss, p₀, LBFGS())
end

γ̲ = 0.005
calibration = DoubleExponentialCalibration(co2tspan, Eₜ, γres.minimizer..., γ̲)

if isinteractive()
    fitfig = plot(t, γ̂ₜ; label="Observed growth rate γ̂ₜ", c=:black, linewidth=2.5, xlabel="Year", ylabel="Growth rate")
    plot!(fitfig, t, t -> γ(t - baselineyear, calibration); label="Fitted growth rate γ(t)", c=:red, linewidth=2.5, linestyle=:dash)
    fitfig
end

# --- Implied emissions
begin
    m₀ = first(m)
    γfn = @closure (m, calibration, t) -> γ(t - baselineyear, calibration)
    γprob = ODEProblem(γfn, m₀, co2tspan, calibration)

    calibratedpath = solve(γprob, Tsit5(), saveat=range(γprob.tspan...; step=1.))

    m₀ = calibratedpath(baselineyear)
    M₀ = Mᵖ * exp(m₀)
    Mₜ = @. Mᵖ * exp(calibratedpath.u)

    @printf "Calibrated error %.2e [p.p.m.]\n" maximum(abs, Mₜ .- co2calibrationdf.Concentration)
end

if isinteractive() # Check simulated fit
    Mfig = plot(co2calibrationdf.Year, Mₜ; c=:black, linestyle=:dash, ylabel=L"Concentration $[\si{\ppm}]$", label=L"Fitted $M^{\textrm{np}}_t$")

    plot!(Mfig, co2calibrationdf.Year, co2calibrationdf.Concentration; c=:black, label=L"SSP5-8.5 $\hat{M}_t$", alpha=0.5)

    error = @. (Mₜ - co2calibrationdf.Concentration) / co2calibrationdf.Concentration

    errorfig = plot(co2calibrationdf.Year, error; c=:black, label=L"Error $M^{\textrm{np}}_t / \hat{M_t} - 1$", xlabel="Year")

    co2errorfig = plot(Mfig, errorfig; layout=(2, 1), size=300 .* (√2, 2), link=:x)

    if SAVEFIG
        savefig(co2errorfig, joinpath(PLOTPATH, "co2error.tikz"))
    end

    co2errorfig
end

# --- Decay rate calibration
if isinteractive() # Check cumulative emissions vs concentration
    gapfig = plot(co2calibrationdf.Year, Mₜ .- Mₜ[1]; c=:black, ylabel=L"Concentration $[\si{\ppm}]$", label=L"$M^{\textrm{np}}_t - M^{\textrm{np}}_{2012}$", xlabel = L"Year $t$")

    plot!(gapfig, co2calibrationdf.Year, cumsum(co2calibrationdf.Emissions); c=:black, linestyle=:dash, label=L"$\int E_t \mathrm{dt}$", alpha=0.5)

    if SAVEFIG
        savefig(gapfig, joinpath(PLOTPATH, "gapfig.tikz"))
    end

    gapfig
end

function saturationdecay(M, p)
    δ₀, α, δ₁, β, Mᶜ, δ̄ = p
    ΔM = M - Mᶜ

    return δ₀ * exp(-α * ΔM) - δ₁ * exp(-β * ΔM) + δ̄
end

begin # Compute decay rate observations
    γ̂ = [γ(t - baselineyear, calibration) for t in co2calibrationdf.Year]
    δ̂ = co2calibrationdf.Emissions ./ Mₜ - γ̂
end

function decayloss(p, optparameters)
    Mₜ, δ̂ = optparameters
    δ = [saturationdecay(M, p) for M in Mₜ]

    return sum(abs2, δ̂ - δ)
end

begin # Solve parameters of saturation decay
    decaylossfn = Optimization.OptimizationFunction(decayloss, AutoForwardDiff());
    optparameters = (Mₜ, δ̂);
    
    # Bounds for double exponential parameters
    p₀ = MVector(0.005, 0.002, 0.007, 0.001, 1100.0, 0.001)  # δ₀, α, δ₁, β, Mᶜ, δ̄
    lb = MVector(0.0, 0., 0.0, 0., 0., -Inf) # δ₀, α, δ₁, β, Mᶜ, δ̄
    ub = MVector(Inf, Inf, Inf, Inf, Inf, 0.002)
    
    decayproblem = Optimization.OptimizationProblem(decaylossfn, p₀, optparameters; lb=lb, ub=ub)
    decaysol = solve(decayproblem, Fminbox(LBFGS()); iterations = 100_000)

    decay = SaturationRecoveryDecay(decaysol.u...)
end

# Check feasibility
δ̲, _ = gssmin(M -> δₘ(M, decay), 400, 3000; tol = 1e-2)
@assert δ̲ + γ̲ > 0

if isinteractive() # Check cumulative emissions vs concentration
    Mspace = range(minimum(Mₜ), 1.2maximum(Mₜ), 202)
    yticks = -0.001:0.001:0.01
    yticklabels = [L"%$(round(100y, digits = 2)) \%" for y in yticks]

    δfig = scatter(Mₜ, δ̂; xlabel = L"M", ylabel = L"Decay rate $\delta$", label = L"Implied decay $\hat{\delta}$", markersize = 2, c = :black, ytick = (yticks, yticklabels), ylims = extrema(yticks))
    plot!(Mspace, M -> δₘ(M, decay) ; c=:black, label = L"Fit $\delta_m(M)$")

    if SAVEFIG
        savefig(δfig, joinpath(PLOTPATH, "delta.tikz"))
    end

    δfig
end

# CALIBRATION OF TEMPERATURE
begin # --- Extract lower and upper bounds for the temperature
    nptemperature = copy(temperature[(npscenario, 0.5)])

    # Temperature without tipping elements
    nptemperature[!, "T lower"] = temperature[(npscenario, 0.05)][:, "T_Uncoupled"]
    nptemperature[!, "T upper"] = temperature[(npscenario, 0.95)][:, "T_Uncoupled"]
    rename!(nptemperature, :T_Uncoupled => "T")

    # Temperature with tipping elements
    nptemperature[!, "T TE lower"] = temperature[(npscenario, 0.05)][:, "T_Coupled"]
    nptemperature[!, "T TE upper"] = temperature[(npscenario, 0.95)][:, "T_Coupled"]
    rename!(nptemperature, :T_Coupled => "T TE")

    select!(nptemperature, Not([:Quantile, :Scenario]))
end;

const Tᵖ = 287.15
const η = 5.67e-8 # Stefan-Boltzmann constant in Wm⁻²K⁻⁴
const S₀ = 235.0 # Incoming radiative forcing

begin
    ghgcalibrationhorizon = 80.
    ghgspan = baselineyear .+ (0, ghgcalibrationhorizon)
    t₀, t₁ = ghgspan

    m̂ = log.(co2calibrationdf[t₀.≤co2calibrationdf.Year.≤t₁, "Concentration"] ./ Mᵖ)
    T̂ = nptemperature[t₀ .≤ nptemperature.Year .≤ t₁, "T"] # Temperature in deviation from pre-industrial level
    X = hcat(ones(length(m̂)), m̂)
    y = @. η * (T̂ + Tᵖ)^4 - S₀
    G₀, G₁ = (X'X) \ (X'y)

    error = sum(abs2, X * [G₀, G₁] - y)

    @printf "G₀ = %.3f, G₁ = %.3f; residual = %.3f \n" G₀ G₁ error
end

begin # Initialize the temperature matching problem
    constants = (η, S₀, G₀, G₁, Tᵖ)
    T̂upper = nptemperature[t₀ .≤ nptemperature.Year .≤ t₁, "T upper"]
    T̂lower = nptemperature[t₀ .≤ nptemperature.Year .≤ t₁, "T lower"]

    T₀ = T̂[1]
    u₀ = SVector(m₀, T₀)

    α₀ = 1.2
    ϵ₀ = 0.15
    σ₀ = 0.01
    p₀ = SVector(ϵ₀, σ₀, α₀)
end;

# --- Optimization of ϵ first
function dTimpulse(T, parameters, t)
    toestimate, constants, impulse = parameters
    ϵ = toestimate[1]
    m = impulse[1]
    
    η, S, G₀, G₁, Tᵖ = constants

    r = S - η * (T + Tᵖ)^4 # Forcing without tipping element
    ghgforcing = G₀ + G₁ * m

    return (r + ghgforcing) / ϵ
end

begin
    m̄ = (η * (T₀ + Tᵖ)^4 - (S₀ + G₀)) / G₁
    Δm = log((M₀ + 47 / 0.75) / Mᵖ) # ≈ 47 p.p.m. increase
    T̄ = find_zero(T -> S₀ - η * (T + Tᵖ)^4 + G₀ + G₁ * Δm, T₀ + Tᵖ)
    impulse = (Δm, T₀, T̄)
end

function distancetohalf(T, t, integrator)
    impulse = integrator.p[end]
    _, T₀, T̄ = impulse

    return (T̄ - T) - (T̄ - T₀) / 2
end

impulseproblem = ODEProblem(dTimpulse, T₀, ghgspan, (p₀, constants, impulse))

halflifeloss = ϵ -> begin
    toestimate = (ϵ, σ₀, α₀)

    sol = solve(impulseproblem, KenCarp4(); p = (toestimate, constants, impulse), callback = ContinuousCallback(distancetohalf, terminate!), save_everystep = false, save_start = false, save_end = false)

    t̄ = (sol.t[end] - baselineyear)

    return SciMLBase.successful_retcode(sol) ? abs2(t̄ - 1 / 4) : Inf
end

_, ϵ = gssmin(halflifeloss, 0.01, 5.)

function Flinear(u, parameters, t)
    toestimate, defaults = parameters
    ϵ = toestimate[1]
    calibration, baselineyear, η, S₀, G₀, G₁, Tᵖ = defaults

    m, T = u
    r = S₀ - η * (T + Tᵖ)^4 # Forcing without tipping element
    ghgforcing = G₀ + G₁ * m

    dm = γ(t - baselineyear, calibration)
    dT = (r + ghgforcing) / ϵ

    return SVector(dm, dT)
end

defaults = (calibration, baselineyear, η, S₀, G₀, G₁, Tᵖ)

if isinteractive() let
    p = SVector(ϵ, σ₀, α₀)
    npmedianprob = ODEProblem(Flinear, u₀, ghgspan, (p, defaults))
    sol = solve(npmedianprob, KenCarp4(); save_idxs = 2, saveat = 1.)

    compfig = plot(xlabel="Year", ylabel=L"Temperature $[\si{\degree}]$", title=L"Comparison: $\hat{T}$ vs fitted $T$ with $\epsilon$", legend=:topleft)

    plot!(compfig, 2020:2100, T̂; label=L"Observed $\hat{T}$", c=:black, linewidth=2.5, alpha=0.7)
    plot!(compfig, 2020:2100, sol.u; label=L"Fitted $T$ with $\epsilon = %$(round(ϵ, digits = 4))$", c=:darkred, linewidth=2.5, linestyle=:dash)

    compfig
end end

# Optimize loss second
function noiselinear(u, parameters, t)
    p = first(parameters)
    ϵ, σ, α = p
    T = u[2]

    return SVector(0., T^α * (σ / ϵ))
end

noiseprob = SDEProblem(Flinear, noiselinear, u₀, ghgspan, ((ϵ, σ₀, α₀), defaults))
ensemblenoiseprob = EnsembleProblem(noiseprob)

function quantileloss(σ, noiseoptparams)
    ensemblenoiseprob, T̂spread = noiseoptparams
    (ϵ, _, α), defaults = ensemblenoiseprob.prob.p

    simparameters = ((ϵ, σ, α), defaults)
    sim = solve(ensemblenoiseprob, ImplicitEM(); p = simparameters, saveat = 1., save_idxs = 2, trajectories = 500)

    if !sim.converged return Inf end
    spread = timestep_quantile(sim, 0.95, :) - timestep_quantile(sim, 0.05, :)

    return sum(abs2, T̂spread - spread)
end

begin
    noiseoptparams = (ensemblenoiseprob, (T̂upper - T̂lower))
    σobjfn = @closure σ -> quantileloss(σ, noiseoptparams) # Tests
    _, σ = gssmin(σobjfn, 0., 1.; tol = 1e-2)
end

if isinteractive() let
    p = SVector(ϵ, σ, α₀)
    parameters = (p, defaults)
    sol = solve(ensemblenoiseprob, ImplicitEM(); p = parameters, save_idxs = 2, saveat = 1., trajectories = 1_000)
    quantiles = timestep_quantile(sol, (0.05, 0.5, 0.95), 1:81)

    years = range(t₀, t₁; step=1.)
    Tlower = @. getindex(quantiles, 1)
    Tmedian = @. getindex(quantiles, 2)
    Tupper = @. getindex(quantiles, 3)

    # Plot observed data with confidence bands
    obsfig = plot(ylims = (1, 8), ylabel="Temperature", xlabel = "Year")

    # Plot fitted data with confidence bands
    plot!(obsfig, years, Tlower; fillrange=Tupper, fillalpha=0.2, linewidth=0., color=:darkred)
    plot!(obsfig, years, Tmedian; color=:darkred)

    plot!(obsfig, years, T̂lower; c = :black, linestyle = :dash)
    plot!(obsfig, years, T̂upper; c = :black, linestyle = :dash)
    plot!(obsfig, years, T̂; c  =:black, linestyle = :dash)

    obsfig
end end

if isinteractive() # Check calibration
    u₀ = SVector(m₀, T₀)
    t₀ = ghgspan[1]
    t₁ = min(ghgspan[2] + 50., nptemperature.Year[end])
    tdx = t₀ .≤ nptemperature.Year .≤ t₁
    checkprob = ODEProblem(Flinear, u₀, (t₀, t₁), (ϵ, defaults))
    sol = solve(checkprob, AutoVern9(Rodas5P()); saveat=t₀:t₁)

    Tfitfig = vspan(collect(ghgspan); alpha=0.4, c=:lightgray, label="Calibration period", ylabel=L"Temperature $[\si{\degree}]$", legend=:topleft, xlims=extrema(sol.t))

    plot!(Tfitfig, nptemperature[tdx, "Year"], nptemperature[tdx, "T"]; label=L"FaIRv2 SSP5-8.5 $\hat{T}_t$", c=:black, linewidth=2.5, alpha=0.5)
    plot!(Tfitfig, nptemperature[tdx, "Year"], nptemperature[tdx, "T lower"]; fillrange=nptemperature[tdx, "T upper"], c=:black, linewidth=0., alpha=0.1)

    T = getindex.(sol.u, 2) # Extract temperature
    plot!(Tfitfig, sol.t, T; label=L"Temperature $T_t$", c=:black, linestyle=:dash, linewidth=2.5)

    error = @. T - nptemperature[tdx, "T"]

    errorfig = hline([0.]; linestyle=:dash, color=:black, linewidth=1.5, ylabel=L"Temperature $[\si{\degree}]$", xlabel=L"Year, $t$", legend=:topright, xlims=extrema(sol.t))

    vspan!(errorfig, collect(ghgspan); alpha=0.4, c=:lightgray)
    plot!(errorfig, sol.t, error; c=:black, ylabel=L"Temperature $[\si{\degree}]$", label=L"Error $T_t - \hat{T_t}$", xlabel=L"Year, $t$", legend=:topright)


    Tcalfig = plot(Tfitfig, errorfig; layout=(2, 1), size=300 .* (√2, 2), link=:x)


    if SAVEFIG
        savefig(Tcalfig, joinpath(PLOTPATH, "temperaturecalibration.tikz"))
    end

    Tcalfig
end

begin # Hogg definition
    hogg = Hogg(
        T₀=T₀, Tᵖ=Tᵖ, M₀=M₀, Mᵖ=Mᵖ,
        S₀=S₀, η=η, ϵ=ϵ, 
        G₀=G₀, G₁=G₁,
        σ=σ, α=α₀
    )

    linearclimate = LinearClimate(hogg, decay)
end

# TIPPING POINT CALIBRATION
if isinteractive()
    excesstfig = plot(xlabel=L"Year, $t$", yaxis="Temperature [°]", xlims=(2020, 2150), legend=:topleft, xticks=2020:20:2150)

    plot!(excesstfig, nptemperature.Year, nptemperature."T TE upper" - nptemperature."T upper"; color=:black, fillrange=nptemperature."T TE lower" - nptemperature."T lower", fillalpha=0.05, linewidth=0., label=false)

    plot!(excesstfig, nptemperature.Year, nptemperature."T TE" - nptemperature."T"; color=:black, linewidth=2.5)

    if SAVEFIG
        savefig(excesstfig, joinpath(PLOTPATH, "excesstfig.tikz"))
    end

    excesstfig
end

# Extract data for calibration
function constructquantiledf(concentration, temperature, scenario; qs=[0.05, 0.5, 0.95])
    lowM, midM, highM = [concentration[(scenario, "carbon_dioxide", q)] for q in qs]

    lowT, midT, highT = [temperature[(scenario, q)] for q in qs]

    lowdf = DataFrame(
        Year=lowM.Year,
        Concentration=lowM.Concentration,
        T_Coupled=lowT.T_Coupled,
        T_Uncoupled=lowT.T_Uncoupled,
        Quantile=lowM.Quantile
    )

    middf = DataFrame(
        Year=midM.Year,
        Concentration=midM.Concentration,
        T_Coupled=midT.T_Coupled,
        T_Uncoupled=midT.T_Uncoupled,
        Quantile=midM.Quantile
    )

    highdf = DataFrame(
        Year=highM.Year,
        Concentration=highM.Concentration,
        T_Coupled=highT.T_Coupled,
        T_Uncoupled=highT.T_Uncoupled,
        Quantile=highM.Quantile
    )

    return lowdf, middf, highdf
end

# Dynamics of the calibration
function coupledsystem(u, parameters, t)
    hogg, calibration, _Tᶜ, _ΔS, _L = parameters
    Tᶜ, ΔS, L = promote(_Tᶜ, _ΔS, _L)

    feedback = Feedback(Tᶜ, ΔS, L)
    baselineyear = calibration.calibrationspan[1]
    m, Tˡ, T = u

    dm = γ(t - baselineyear, calibration)
    
    f = Model.ghgforcing(m, hogg)
    dTˡ = (f + Model.radiativeforcing(Tˡ, hogg)) / hogg.ϵ
    dT = (f + Model.radiativeforcing(T, hogg) + λ(T, feedback)) / hogg.ϵ

    return SVector(dm, dTˡ, dT)
end

tedfs = constructquantiledf(concentration, temperature, npscenario; qs=[0.05, 0.5, 0.95])

function tippingelementloss(p, optparameters)
    coupledprob, ΔTtrajectory = optparameters
    Tᶜ, ΔS, L = p
    hogg, calibration = coupledprob.p[1:2]

    parameters = (hogg, calibration, Tᶜ, ΔS, L)
    sol = solve(coupledprob, RadauIIA5(); p=parameters, saveat=1., verbose=false, reltol=1e-8)

    if SciMLBase.successful_retcode(sol)
        ΔT = @. getindex(sol.u, 3) - getindex(sol.u, 2)

        return sum(abs2, ΔT - ΔTtrajectory)
    else
        return Inf
    end
end

function constraints(p, optparameters)
    res = Vector{Float64}(undef, 4)
    return constraints!(res, p, optparameters)
end
function constraints!(res, p, optparameters)
    coupledprob = first(optparameters)
    hogg = first(coupledprob.p)

    Tᶜ, ΔS, L = p

    res[1] = (2L * ΔS / 3) + 4 * Model.radiativeforcing′(Tᶜ, hogg) # > 0 Fold constraint

    # > 0 Parameter constraints
    res[2] = Tᶜ
    res[3] = L
    res[4] = ΔS

    return res
end

begin # Calibrate adjustment speed
    u₀ = SVector(m₀, T₀, T₀)

    tecalibrationhorizon = 100. + baselineyear
    centurydx = searchsortedfirst(co2calibrationdf.Year, tecalibrationhorizon)
    mtarget = log(co2calibrationdf[centurydx, "Concentration"] / Mᵖ)

    Tᶜ₀ = 3.
    L₀ = 7.
    ΔS₀ = 10.
    p₀ = [Tᶜ₀, ΔS₀, L₀]

    adtype = SecondOrder(AutoForwardDiff(), AutoForwardDiff())
    lcons = [0., 0., 0., 0.]
    ucons = [Inf, Inf, Inf, Inf]

    feedbacks = Feedback{Float64}[]
    for (i, df) in enumerate(tedfs)
        subdf = df[baselineyear .≤ df.Year .≤ tecalibrationhorizon, :]

        parameters = (hogg, calibration, Tᶜ₀, ΔS₀, L₀)
        coupledprob = ODEProblem(coupledsystem, u₀, extrema(subdf.Year), parameters)

        ΔTtrajectory = subdf.T_Coupled .- subdf.T_Uncoupled

        objfunction = Optimization.OptimizationFunction(tippingelementloss, adtype; cons=constraints!)
        optproblem = Optimization.OptimizationProblem(objfunction, p₀, (coupledprob, ΔTtrajectory); lcons, ucons)

        teresult = solve(optproblem, IPNewton(); iterations=10_000)
        Tᶜ, ΔS, L = teresult.u

        if !SciMLBase.successful_retcode(teresult)
            @warn @sprintf "Result %i not converged.\n" i
        else
            @printf "Problem %i converged with ΔTᶜ=%.3f, ΔS=%.3f, L=%.3f, error=%.3e.\n" i Tᶜ ΔS L teresult.objective
        end

        feedback = Feedback(Tᶜ, ΔS, L)

        push!(feedbacks, feedback)
    end
end

function extendedcoupledsystem!(du, u, parameters, t)
    linearclimate, calibration, feedbacks = parameters
    m = u[1]
    T = @view u[2:end]
    dT = @view du[2:end]
    baselineyear = calibration.calibrationspan[1]

    du[1] = γ(t - baselineyear, calibration)
    dT[1] = μ(T[1], m, linearclimate) / linearclimate.hogg.ϵ

    for (i, feedback) in enumerate(feedbacks)
        tippingclimate = TippingClimate(linearclimate.hogg, linearclimate.decay, feedback)
        dT[i + 1] = μ(T[i + 1], m, tippingclimate) / linearclimate.hogg.ϵ
    end
end;

if isinteractive()
    u₀ = [m₀, T₀, T₀, T₀, T₀]
    parameters = (linearclimate, calibration, feedbacks)

    extenedcoupledprob = ODEProblem(extendedcoupledsystem!, u₀, (baselineyear, tecalibrationhorizon), parameters)

    sol = solve(extenedcoupledprob, RadauIIA5(); saveat=1., reltol=1e-8)

    solfig = plot(xlabel="Year", ylabel=L"Temperature $[\si{\degree}]$", title="Temperature Dynamics with Tipping Element", legendtitle="Quantile")

    plot!(solfig, sol.t, getindex.(sol.u, 2); c=:black, label="Linear")

    qs = (0.05, 0.5, 0.95)
    c = palette(:reds, 3)
    for i in 1:3
        plot!(solfig, sol.t, getindex.(sol.u, i + 2); labels=qs[i], c=c[i])
    end

    solfig
end

# Check Nullclines
if isinteractive()
    Tspace = range(hogg.T₀, hogg.T₀ + 6.; length=101)
    nullclinefig = plot(ylabel="Temperature [°]", xlabel="CO2e concentration (ppm)")

    for (i, feedback) in enumerate(feedbacks)
        tippingclimate = TippingClimate(linearclimate.hogg, linearclimate.decay, feedback)
        mnullcline = [ mstable(T, tippingclimate) for T in Tspace ]
        plot!(nullclinefig, mnullcline, Tspace; label=qs[i], c=c[i])
    end

    nullclinefig
end


# Save outcome of calibration in file
jldopen(joinpath(calibrationpath, "climate.jld2"), "w+") do file
    feedbacklower, feedback, feedbackhigher = feedbacks
    @pack! file = hogg, calibration, feedbacklower, feedback, feedbackhigher, decay
end