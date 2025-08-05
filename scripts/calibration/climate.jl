using Revise

using DotEnv, UnPack
using CSV, DataFrames, JLD2
using StaticArrays
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis
using DiffEqParamEstim, Optimization, OptimizationOptimJL, OptimizationPolyalgorithms
using Roots, FastClosures
using Statistics, LinearAlgebra

using Plots, Printf, PGFPlotsX, Colors, ColorSchemes, LaTeXStrings
pgfplotsx()

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{siunitx}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\ppm}{p.p.m.}")

default(label = false, dpi = 180, linewidth = 2.5)

using Model, Grid

includet("../utils/simulating.jl")

PLOTPATH = "papers/job-market-paper/submission/plots"
DATAPATH = "data"
SAVEFIG = false
PALETTE = colorschemes[:grays]
RUNESTIMATE = false
calibrationpath = joinpath(DATAPATH, "calibration.jld2")

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
function loadhierdataframe(variablepathpair::Pair{K, String}; groupkeys = [:Group]) where K <: Union{String, Symbol}
    ngroups = length(groupkeys)
    variablename, filepath = variablepathpair

    df = DataFrame(CSV.File(filepath))

    years = parse.(Int, df[(2 + ngroups):end, 1])
    nyears = length(years)
    
    scenarios = map(parsescenario, names(df)[2:end])
    nscenarios = length(scenarios)
    
    variables = [
        groupkey => repeat(collect(df[i, 2:end]), inner = nyears)
        for (i, groupkey) in enumerate(groupkeys)
    ]

    data = safeparse.(Float64, df[(2 + ngroups):end, 2:end] |> Matrix)
    vecdata = vec(data)

    newdf = DataFrame(
        :Year => repeat(years, outer = nscenarios),
        :Scenario => repeat(scenarios, inner = nyears),
        variables...,
        Symbol(variablename) => safeparse.(Float64, vecdata)
    )

    return newdf
end

coupleddir = joinpath(DATAPATH, "deutloff", "model_output", "coupled_ensemble"); @assert isdir(coupleddir)
uncoupleddir = joinpath(DATAPATH, "deutloff", "model_output", "uncoupled_ensemble"); @assert isdir(uncoupleddir)

probabilities = groupby(
    loadhierdataframe(:Probability => joinpath(coupleddir, "tip_prob_total.csv"); groupkeys = [:TE]),
    [:Scenario, :TE]
)

begin # Load temperature dataframe
    temperaturedf = loadhierdataframe(:T_Coupled => joinpath(coupleddir, "T.csv"); groupkeys = [:Quantile]);
    temperaturedf.T_Uncoupled = loadhierdataframe(:T_Uncoupled => joinpath(uncoupleddir, "T.csv"); groupkeys = [:Quantile]).T_Uncoupled

    temperature = groupby(temperaturedf, [:Scenario, :Quantile])
end;

begin # Load emissions and GHGs dataframes
    concentrationdf = loadhierdataframe("Concentration" => joinpath(coupleddir, "C.csv"); groupkeys = [:Particle, :Quantile])
    concentrationdf.Quantile = safeparse.(Float64, concentrationdf.Quantile)

    emissionsdf = loadhierdataframe("Emissions" => joinpath(coupleddir, "Emm.csv"); groupkeys = [:Particle, :Quantile])
    emissionsdf.Quantile = safeparse.(Float64, emissionsdf.Quantile)

    concentrationdf.Emissions = emissionsdf.Emissions
    concentration = groupby(concentrationdf, [:Scenario, :Particle, :Quantile])
end;

let # Figure CO₂ concentration
    qs = [0.05, 0.5, 0.95];
    figscenarios = ["SSP1 1.9", "SSP2 4.5", "SSP4 3.4", "SSP5 8.5"]
    cmap = palette(:viridis, length(figscenarios); rev = true)

    co2fig = plot(xlabel = "Year", yaxis = "Concentration (ppm)", title = "Carbon Dioxide Concentration", xlims = (2000, 2400))
    excesstfig = plot(xlabel = "Year", yaxis = "Temperature [°]", title = "Additional temperature", xlims = (2012, 2400))
    
    for (i, scenario) in enumerate(figscenarios)
        co2 = Tuple(concentration[(scenario, "carbon_dioxide", q)] for q in qs)
        temp = Tuple(temperature[(scenario, q)] for q in qs)

        c = cmap[i]
        
        plot!(co2fig, co2[1].Year, co2[1].Concentration; color = c, fillrange = co2[3].Concentration, fillalpha = 0.05, linewidth = 0., label = false)
        plot!(co2fig, co2[2].Year, co2[2].Concentration; label = scenario, color = c, linewidth = 2.5)

        plot!(excesstfig, temp[1].Year, temp[1].T_Coupled - temp[1].T_Uncoupled; color = c, fillrange = temp[3].T_Coupled - temp[3].T_Uncoupled, fillalpha = 0.05, linewidth = 0., label = false)
        plot!(excesstfig, temp[2].Year, temp[2].T_Coupled - temp[2].T_Uncoupled; label = scenario, color = c, linewidth = 2.5)
    end

    impulsefig = plot(co2fig, excesstfig; layout = (1, 2), size = 600 .* (2√2, 1), margins = 5Plots.mm, legend = :topleft)
end

# CALIBRATION OF CO₂ GROWTH RATE
const npscenario = "SSP5 8.5"

# -- Construct CO2 equivalent concentration
function computeco2equivalence(concentration, quantile, gwp)
    co2equivalence = deepcopy(concentration[(npscenario, "carbon_dioxide", quantile)])
    co2equivalence[!, "CO2 Concentration"] .= co2equivalence[:, "Concentration"] 
    co2equivalence[!, "CO2 Emissions"] .= co2equivalence[:, "Emissions"] 

    for (particle, (gwpvalue, factor)) in gwp
        df = concentration[(npscenario, particle, quantile)]
        
        co2equivalence[!, "$particle Concentration"] .= df.Concentration .* factor * gwpvalue
        co2equivalence[!, "Concentration"] .+= co2equivalence[!, "$particle Concentration"]

        co2equivalence[!, "$particle Emissions"] .= df.Emissions .* gwpvalue
        co2equivalence[!, "Emissions"] .+= co2equivalence[!, "$particle Emissions"]
    end

    return co2equivalence
end

begin
    gwp = Dict(
        "methane" => (29.8, 1e-3), 
        "nitrous_oxide" => (273., 1e-3),
        "nf3" => (17_400., 1e-6),
        "sf6" => (24_300., 1e-6), 
    )

    co2equivalence = computeco2equivalence(concentration, 0.5, gwp)
    co2equivalencelower = computeco2equivalence(concentration, 0.05, gwp)
    co2equivalenceupper = computeco2equivalence(concentration, 0.95, gwp)
end;

let # Figure CO₂ concentration in the no policy scenario
    co2equivalencefig = plot(xlabel = "Year", yaxis = "Concentration (ppm)", title = "Carbon Dioxide Concentration in SSP5 8.5", xlims = (1900, 2150))

    concentrationnames = filter(m -> occursin("Concentration", m), names(co2equivalence))
    colors = palette(:tab10, length(concentrationnames); rev = true)

    for (i, cname) in enumerate(concentrationnames)
        color = colors[i]
        plot!(co2equivalencefig, co2equivalence.Year, co2equivalence[:, cname]; label = cname, linewidth = 2.5, c = color)
        plot!(co2equivalencefig, co2equivalencelower.Year, co2equivalencelower[:, cname]; label = false, fillrange = co2equivalenceupper[:, cname] .+ 1e-3, fillalpha = 0.2, linewidth = 0., c = color)
    end

    co2equivalencefig
end

let # Figure fraction of forcing
    yearbound = (1980, 2200)
    tdxs = @. yearbound[1] ≤ co2equivalence.Year ≤ yearbound[2]
    
    xtick = collect(range(yearbound...; step = 20))
    xticklabelposition = range(yearbound...; step = 40)
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
        title = raw"Fraction of $\mathrm{CO}_2$ to $\mathrm{CO}_2$-e concentration",
        ylabel_style = {align = "center"},
        xmin = yearbound[1], xmax = yearbound[2],
        xtick = xtick, xticklabels = xticklabels,
        ytick = ytick, yticklabels = yticklabels,
        ymin = 0.69, ymax = 1.0,
    })

    curve = @pgf Plot({
        line_width = 2.5
    }, Coordinates(co2equivalence.Year[tdxs], fracradiation))

    push!(fracfig, curve)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "fracfig.tikz"), fracfig; include_preamble=true)
    end

    fracfig
end

# --- Computing parametric emissions form
begin # Setup CO₂e maximisation problem
    baselineyear = 2020.; τ = 2500. - baselineyear;
    co2tspan = baselineyear .+ (0., τ);

    tdxs = baselineyear .≤ co2equivalence.Year .≤ (baselineyear + τ);
    co2calibrationdf = co2equivalence[tdxs, :]

    Mᵖ = mean(co2equivalence[1800 .≤ co2equivalence.Year .≤ 1900, "Concentration"])
    m = @. log(co2calibrationdf.Concentration / Mᵖ)
    m₀ = m[1]
end;

"Growth rate of emissions in the form of a parametric function `γₜ : baselineyear + [0, τ] -> [0, ∞)`."
function parametricemissions(m, parameters, x)
    p, defaults = parameters
    baselineyear = first(defaults)
    t = x - baselineyear # Shift as t starts at baselineyear
    
    return γ(t, Tuple(p))
end

function concentrationloss(p, optparams)
    concentrationprob, defaults, Mᵖ, targetM, α = optparams

    t₀, t₁ = concentrationprob.tspan
    sol = solve(concentrationprob, Tsit5(); p = (p, defaults), saveat = t₀:t₁)

    M = @. Mᵖ * exp(sol.u)
    weights = @. exp(-α * (0:(t₁ - t₀)))

    return sum(abs2, @. weights * (M - targetM))
end

begin # Initialize the CO₂e calibration problem
    p₀ = MVector{6}(0.0003, 0.02, 0.002, 0.005, 0.002, 1e-6)
    defaults = (baselineyear, );

    concentrationprob = ODEProblem(parametricemissions, m₀, co2tspan, (p₀, defaults)); solve(concentrationprob, Tsit5())

    α = 0.005
    optparams = (concentrationprob, defaults, Mᵖ, co2calibrationdf.Concentration, α);
    concentrationlossfunction = Optimization.OptimizationFunction(concentrationloss, Optimization.AutoForwardDiff());
    optprob = Optimization.OptimizationProblem(concentrationlossfunction, p₀, optparams)
    result = solve(optprob, PolyOpt(); iterations = 100_000)
end;

# --- Implied emissions
begin
    γparameters = Tuple(result.u)
    γfn = @closure (m, γparameters, t) -> γ(t - baselineyear, γparameters)
    γprob = ODEProblem(γfn, m₀, co2tspan, γparameters)

    calibratedpath = solve(γprob, Tsit5(), saveat = range(γprob.tspan...; step = 1.))

    @printf "Calibrated error %e\n" maximum(abs, calibratedpath.u .- m);

    m₀ = calibratedpath(baselineyear)
    M₀ = Mᵖ * exp(m₀)
    Mₜ = Mᵖ * exp.(calibratedpath.u)

    ppmoverGt = 2.13 * 3.664
    Eₜ = diff(Mₜ) * ppmoverGt

    calibration = Calibration(baselineyear, Eₜ, γparameters, τ)
end

begin # Check simulated fit
    Mfig = plot(co2calibrationdf.Year, Mₜ; c = :black, linestyle = :dash, ylabel = L"Concentration $[\si{\ppm}]$", label = L"Fitted $M_t$")

    plot!(Mfig, co2calibrationdf.Year, co2calibrationdf.Concentration; c = :black, label = L"SSP5-8.5 $\hat{M}_t$", alpha = 0.5)

    error = @. (Mₜ - co2calibrationdf.Concentration)

    errorfig = plot(co2calibrationdf.Year, error; c = :black, ylabel = L"Concentration $[\si{\ppm}]$", label = L"Error $M_t - \hat{M_t} \; [\si{\ppm}]$", xlabel = "Year")

    co2errorfig = plot(Mfig, errorfig; layout = (2, 1), size = 300 .* (√2, 2),legend = :bottomright, link = :x)
    
    if SAVEFIG
        savefig(co2errorfig, joinpath(PLOTPATH, "co2error.tikz"))
    end

    co2errorfig
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

const kelvintocelsius = 273.15;
const deviationtokelvin = 287.15;

begin
    η = 5.67e-8 # Stefan-Boltzmann constant in Wm⁻²K⁻⁴
    S = 239.75 # Incoming radiative forcing

    ghgcalibrationhorizon = 80.
    ghgspan = baselineyear .+ (0, ghgcalibrationhorizon)
    t₀, t₁ = ghgspan

    m̂ = log.(co2calibrationdf[t₀ .≤ co2calibrationdf.Year .≤ t₁, "Concentration"] ./ Mᵖ)
    T̂ = nptemperature[t₀ .≤ nptemperature.Year .≤ t₁, "T"] .+ deviationtokelvin # Temperature in Kelvin

    X = hcat(ones(length(m̂)), m̂)
    y = η * T̂.^4 .- S
    G₀, G₁ = (X'X) \ (X'y)

    error = sum(abs2, X * [G₀, G₁] - y)

    @printf "G₀ = %.3f, G₁ = %.3f; residual = %.3f \n" G₀ G₁ error
end

function Flinear(u, parameters, t)
    ϵ, defaults = parameters
    calibration, baselineyear, η, S, G₀, G₁ = defaults

    m, T = u
    r = S - η * T^4 # Forcing with tipping element
    ghgforcing = G₀ + G₁ * m

    dm = γ(t - baselineyear, calibration)
    dT = (r + ghgforcing) / ϵ

    return SVector(dm, dT)
end

# --- Define the loss function
function temperatureloss(ϵ, optparams)
    linearprob, defaults, matchtemperature = optparams

    t₀, t₁ = linearprob.tspan
    sol = solve(linearprob, AutoVern9(Rodas5P()); p = (ϵ, defaults), saveat = t₀:t₁)
    
    if SciMLBase.successful_retcode(sol.retcode)
        T = getindex.(sol.u, 2) .- deviationtokelvin
        return sum(abs2, @. T - matchtemperature)
    else
        return Inf
    end
end

begin # Initialize the temperature matching problem 
    defaults = (calibration, Float64(baselineyear), η, S, G₀, G₁)
    
    T₀ = 1.4 + deviationtokelvin
    u₀ = SVector{2}(m₀, T₀)
    ϵ₀ = 0.15
    linearprob = ODEProblem(Flinear, u₀, ghgspan, (ϵ₀, defaults))
    testsol = solve(linearprob, AutoVern9(Rodas5P()); saveat = t₀:t₁) # Ensure the problem is well-posed
    
    tdx = t₀ .≤ nptemperature.Year .≤ t₁;
    matchtemperature = nptemperature[tdx, "T"]

    optparams = (linearprob, defaults, matchtemperature);

    lossfn = @closure ϵ -> temperatureloss(ϵ, optparams)

    _, ϵ = gssmin(lossfn, 0., 100.)
end

begin # Check calibration
    t₀ = ghgspan[1]
    t₁ = min(ghgspan[2] + 50., nptemperature.Year[end])
    tdx = t₀ .≤ nptemperature.Year .≤ t₁
    checkprob = ODEProblem(Flinear, u₀, (t₀, t₁), (ϵ, defaults))
    sol = solve(checkprob, AutoVern9(Rodas5P()); saveat = t₀:t₁)
    
    Tfitfig = vspan(collect(ghgspan); alpha = 0.4, c = :lightgray, label = "Calibration period", ylabel = L"Temperature $[\si{\degree}]$", legend = :topleft, xlims = extrema(sol.t))

    plot!(Tfitfig, nptemperature[tdx, "Year"], nptemperature[tdx, "T"]; label = L"FaIRv2 SSP5-8.5 $\hat{T}_t$", c = :black, linewidth = 2.5, alpha = 0.5)
    plot!(Tfitfig, nptemperature[tdx, "Year"], nptemperature[tdx, "T lower"]; fillrange = nptemperature[tdx, "T upper"], c = :black, linewidth = 0., alpha = 0.1)

    T = getindex.(sol.u, 2) .- deviationtokelvin; # Extract temperature
    plot!(Tfitfig, sol.t, T; label = L"Temperature $T_t$", c =:black, linestyle = :dash, linewidth = 2.5)

    error = @. T - nptemperature[tdx, "T"]

    errorfig = hline([0.]; linestyle = :dash, color = :black, linewidth = 1.5, ylabel = L"Temperature $[\si{\degree}]$", xlabel = L"Year, $t$", legend = :topright, xlims = extrema(sol.t))
    
    vspan!(errorfig, collect(ghgspan); alpha = 0.4, c = :lightgray)
    plot!(errorfig, sol.t, error; c = :black, ylabel = L"Temperature $[\si{\degree}]$", label = L"Error $T_t - \hat{T_t}$", xlabel = L"Year, $t$", legend = :topright)


    Tcalfig = plot(Tfitfig, errorfig; layout = (2, 1), size = 300 .* (√2, 2), link = :x)


    if SAVEFIG
        savefig(Tcalfig, joinpath(PLOTPATH, "temperaturecalibration.tikz"))
    end

    Tcalfig
end

begin # Hogg calibration
    todaydx = findfirst(co2equivalence.Year .== baselineyear)
    frac = co2equivalence[todaydx, "CO2 Concentration"] ./ co2equivalence[todaydx, "Concentration"]
    N₀ = 286.65543 / frac[1]

    hogg = Hogg(
        T₀ = T₀, M₀ = M₀, Mᵖ = Mᵖ, N₀ = N₀, 
        G₀ = G₀, G₁ = G₁,
    )
end

# TIPPING POINT CALIBRATION
begin
    excesstfig = plot(xlabel = L"Year, $t$", yaxis = "Temperature [°]", xlims = (2020, 2150), legend = :topleft, xticks = 2020:20:2150)

    plot!(excesstfig, nptemperature.Year, nptemperature."T TE upper" - nptemperature."T upper"; color = :black, fillrange = nptemperature."T TE lower" - nptemperature."T lower", fillalpha = 0.05, linewidth = 0., label = false)

    plot!(excesstfig, nptemperature.Year, nptemperature."T TE" - nptemperature."T";  color = :black, linewidth = 2.5)

    if SAVEFIG
        savefig(excesstfig, joinpath(PLOTPATH, "excesstfig.tikz"))
    end

    excesstfig
end

# Extract data for calibration
function constructquantiledf(concentration, temperature, scenario; qs = [0.05, 0.5, 0.95])
    lowM, midM, highM = [concentration[(scenario, "carbon_dioxide", q)] for q in qs]
    
    lowT, midT, highT = [temperature[(scenario, q)] for q in qs]

    lowdf = DataFrame(
        Year = lowM.Year,
        Concentration = lowM.Concentration,
        T_Coupled = lowT.T_Coupled,
        T_Uncoupled = lowT.T_Uncoupled,
        Quantile = lowM.Quantile
    )

    middf = DataFrame(
        Year = midM.Year,
        Concentration = midM.Concentration,
        T_Coupled = midT.T_Coupled,
        T_Uncoupled = midT.T_Uncoupled,
        Quantile = midM.Quantile
    )

    highdf = DataFrame(
        Year = highM.Year,
        Concentration = highM.Concentration,
        T_Coupled = highT.T_Coupled,
        T_Uncoupled = highT.T_Uncoupled,
        Quantile = highM.Quantile
    )

    return lowdf, middf, highdf
end

const modeldfs = constructquantiledf(concentration, temperature, npscenario; qs = [0.05, 0.5, 0.95])

# Dynamics of the calibration
function system!(du, u, albedo, t)
    m, T, Tˡ = u

    # Forcing with tipping element
    T₁ = albedo.Tᶜ + hogg.Tᵖ
    T₂ = T₁ + albedo.ΔT
    inflexion = (T₁ + T₂) / 2

    Lₜ = Model.sigmoid(T - inflexion, albedo.β) 
    λₜ = albedo.λ₁ - albedo.Δλ * Lₜ
    r = hogg.S₀ * (1 - λₜ) - hogg.η * T^4

    # Forcing with no tipping element
    rˡ = Model.radiativeforcing(Tˡ, hogg)

    # GHG forcing
    g = Model.ghgforcing(m, hogg)
    du[1] = γ(t - baselineyear, calibration)
    du[2] = (g + r) / hogg.ϵ
    du[3] = (g + rˡ) / hogg.ϵ
end

u₀ = [log(hogg.M₀ / hogg.Mᵖ), hogg.T₀, hogg.T₀]; # Initial conditions
albedo = Albedo(Tᶜ = 2.5)
tspan = (2020., 2150.); # Time span in years

du = similar(u₀); # Allocate memory for the derivative
system!(du, u₀, albedo, 2020.); # Call the system to ensure it works
prob = ODEProblem(system!, u₀, tspan, albedo);
probsol = solve(prob, AutoVern9(Rosenbrock23()))

mediandf = modeldfs[2];
row = mediandf[mediandf.Year .== 2100, :];
const targetΔT = row.T_Coupled[1] - row.T_Uncoupled[1]

function terminalΔTloss(sim)::Float64
    t = last(tspan)
    _, T, Tˡ = sim(t)
    ΔT = T - Tˡ

    return abs2(targetΔT - ΔT)
end

lossbyparam = @closure Δλ -> begin
    albedo = Albedo(Tᶜ = 2., Δλ = Δλ)
    sol = solve(prob, AutoVern9(Rosenbrock23()); p = albedo)

    return terminalΔTloss(sol)
end

_, Δλ = gssmin(lossbyparam, 0., 0.1)
plot(0:0.001:0.05, lossbyparam; ylabel = "Loss", xlabel = L"\Delta\lambda", linewidth = 2, c = :black, xlims = (0, 0.05))

sol = solve(prob, AutoVern9(Rosenbrock23()); p = Albedo(Tᶜ = 2.0, Δλ = Δλ))

begin
    solfig = plot(xlabel = "Year", ylabel = L"Temperature $[\si{\degree}]$", title = "Temperature Dynamics with Tipping Element", xticks = (0:10:80, 2020:10:2100))

    plot!(solfig, sol; idxs = [2, 3], labels = ["T" "Tˡ"], margins = 5Plots.mm, c = [:darkred :darkblue], linewidth = 2.5)
end

function nullclines(T, albedo)
    @unpack Δλ, ΔT, λ₁, Tᶜ, β = albedo

    # Forcing with tipping element
    T₁ = Tᶜ + hogg.Tᵖ
    T₂ = T₁ + ΔT
    inflexion = (T₁ + T₂) / 2

    Lₜ = Model.sigmoid(T - inflexion, β)
    λₜ = λ₁ - Δλ * Lₜ
    r = hogg.S₀ * (1 - λₜ) - hogg.η * T^4
    m = Model.ghgforcing⁻¹(-r, hogg)

    # Forcing with no tipping element
    rˡ = Model.radiativeforcing(T, hogg)
    mˡ = Model.ghgforcing⁻¹(-rˡ, hogg)
    
    return m, mˡ
end

let
    albedo = Albedo(Tᶜ = 2.0, Δλ = Δλ)

    Tspace = range(0, 6, length = 100) .+ hogg.Tᵖ
    mstable = [nullclines(T, albedo) for T in Tspace]
    
    mlabels = range(280, 1200; length = 5)
    mticks = log.(mlabels)

    Tticks = range(0, 6, length = 10) .+ hogg.Tᵖ
    Tlabels = [Printf.@sprintf("%.1f", T - hogg.Tᵖ) for T in Tticks]

    nullclinefig = plot(xlabel = "CO2 concentration [p.p.m.]", xticks = (mticks, mlabels), ylabel = L"Temperature $[\si{\degree}]$", yticks = (Tticks, Tlabels))

    plot!(nullclinefig, last.(mstable), Tspace; label = "With tipping element", color = :darkred, linewidth = 2.5, linestyle = :dash)
    plot!(nullclinefig, first.(mstable), Tspace; label = "Without tipping element", color = :darkblue, linewidth = 2.5, linestyle = :dash)
end

function noise!(Σ, u, albedo, t)
    # Noise is not used in this model
    Σ[1] = 0.0; 
    Σ[2] = Σ[3] = 0.0999744768;
end


function adjspeed(sim)
    albedo = sim.prob.p
    T₀ = albedo.Tᶜ + 287.15
    
    t₀ = find_zero(t -> sim(t)[2] - T₀, sim.prob.tspan)
    
    timepost = range(t₀, sim.prob.tspan[end]; step = 1/12)
    trajectory = sim(timepost).u
    warming = getindex.(trajectory, 2) .- getindex.(trajectory, 3)
    
    tdx = findfirst(ΔTₜ -> ΔTₜ > 1., warming)
    return timepost[tdx] - t₀
end

function terminalwarming(sim)
    _, T, Tˡ = sim.u[end]
    return T - Tˡ
end

sdeprob = SDEProblem(system!, noise!, u₀, tspan, Albedo(Tᶜ = 2.5));
ensemble = EnsembleProblem(sdeprob);
simulations = solve(ensemble, ImplicitRKMil(); trajectories = 1_000)
speeds = map(adjspeed, simulations)
warmings = map(terminalwarming, simulations)

begin
    speedhistfig = histogram(speeds; bins = 60, linewidth = 1, c = :white,  normalize = true, xlims = (0, Inf), xlabel = "Years", ylabel = "Frequency")

    if SAVEFIG
        savefig(speedhistfig, joinpath(PLOTPATH, "speedhistfig.tikz"))
    end

    speedhistfig
end

begin
    warminghistfig = histogram(warmings; bins = 50, linewidth = 1, c = :white,  normalize = true, xlims = (0, Inf), ylabel = "Frequency")

    if SAVEFIG
        savefig(warminghistfig, joinpath(PLOTPATH, "warminghistfig.tikz"))
    end

    warminghistfig
end

# Save outcome of calibration in file

jldopen(calibrationpath, "w+") do file
    @pack! file = hogg, calibration, albedo
end