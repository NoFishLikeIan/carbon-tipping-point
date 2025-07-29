using Revise

using DotEnv, UnPack
using CSV, DataFrames, JLD2
using StaticArrays
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis
using DiffEqParamEstim, Optimization, OptimizationOptimJL, OptimizationPolyalgorithms
using Roots, FastClosures
using Statistics

using Plots, Printf, PGFPlotsX, Colors, ColorSchemes, LaTeXStrings
pgfplotsx()

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{siunitx}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\ppm}{p.p.m.}")

default(label = false, dpi = 180, linewidth = 2.5)

using Model, Grid

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
const ppmtoGt = 2.13 * 3.664;
baselineyear = 2020;
τ = 140.;
co2calibrationend = baselineyear + τ;

co2calibrationtime = baselineyear:co2calibrationend;
tdxs = baselineyear .≤ co2equivalence.Year .≤ co2calibrationend;
calibrationdf = co2equivalence[tdxs, :]

Mᵖ = mean(co2equivalence[1800 .≤ co2equivalence.Year .≤ 1900, "Concentration"])
m = log.(calibrationdf.Concentration ./ Mᵖ)

function parametricemissions(m, p, t)
    order = length(p) - 1
    γ = zero(m)
    for n in 0:order
        γ += p[n + 1] * t^n
    end

    return γ
end

begin k = 4
    degree = k + 1; # Degree of the polynomial
    p₀ = zeros(degree); # Initial guess for the parameters
    co2calibrationprob = ODEProblem{false}(parametricemissions, m[1], extrema(co2calibrationtime), p₀); solve(co2calibrationprob, Tsit5())

    loss = build_loss_objective(co2calibrationprob, Tsit5(), L2Loss(co2calibrationtime, m), Optimization.AutoForwardDiff(); maxiters = 1_000_000, saveat = co2calibrationtime)

    optprob = Optimization.OptimizationProblem(loss, p₀)
    sol = solve(optprob, BFGS()); p̂ = sol.u;

    calibratedpath = solve(co2calibrationprob, Tsit5(); p = p̂);
    @printf "Calibrated error %e\n" maximum(abs, calibratedpath(co2calibrationtime) .- m);
    γparameters = Tuple(sol.u)
end;

# --- Implied emissions
begin
    m₀ = calibratedpath(baselineyear)
    M₀ = Mᵖ * exp(m₀)
    Mₜ = Mᵖ * exp.(calibratedpath(co2calibrationtime).u)
    Eₜ = diff(Mₜ) * ppmtoGt
    calibration = Calibration{degree}(baselineyear, Eₜ, γparameters, τ)
end

let # Check simulated fit
    parametricemissions(m, p::Calibration, t) = γ(t - baselineyear, p)
    prob = ODEProblem{false}(parametricemissions, m[1], extrema(co2calibrationtime), calibration)
    
    sol = solve(prob, Tsit5())
    Msimulated =  Mᵖ * exp.(sol(co2calibrationtime).u)

    co2fig = plot(co2calibrationtime, Mₜ; label = "Fitted ", c = :darkred, alpha = 0.5)
    plot!(co2fig, co2calibrationtime, Msimulated; label = "Simulated", c = :darkblue, alpha = 0.5)
    plot!(co2fig, co2calibrationtime, calibrationdf.Concentration; label = "True", c = :black)

    γfig = plot(co2calibrationtime, t -> γ(t - baselineyear, calibration); c = :black)

    plot(γfig, co2fig; size = 500 .* (2√2, 1))
end

# CALIBRATION OF TEMPERATURE
begin # --- Extract lower and upper bounds for the temperature
    tdxs = baselineyear .≤ temperature[1].Year
    nptemperature = copy(temperature[(npscenario, 0.5)][tdxs, :])
    
    # Temperature without tipping elements
    nptemperature[!, "T lower"] = temperature[(npscenario, 0.05)][tdxs, "T_Uncoupled"]
    nptemperature[!, "T upper"] = temperature[(npscenario, 0.95)][tdxs, "T_Uncoupled"]
    rename!(nptemperature, :T_Uncoupled => "T")
    
    # Temperature with tipping elements
    nptemperature[!, "T TE lower"] = temperature[(npscenario, 0.05)][tdxs, "T_Coupled"]
    nptemperature[!, "T TE upper"] = temperature[(npscenario, 0.95)][tdxs, "T_Coupled"]
    rename!(nptemperature, :T_Coupled => "T TE")

    select!(nptemperature, Not([:Quantile, :Scenario]))
end

const ϵ = 15.844043907014475;
const kelvintocelsius = 273.15;
const deviationtokelvin = 287.15;

function Fnp!(du, u, parameters, t)
    m, T = u
    G₀, G₁ = parameters
    
    # Forcing with tipping element
    λ₁ = 0.31; S₀ = 340.5; η = 5.67e-8;
    r = S₀ * (1 - λ₁) - η * T^4

    # Greenhouse gas forcing
    G = G₀ + G₁ * m

    du[1] = γ(t - baselineyear, calibration)
    du[2] = (r + G) / ϵ
end

T₀ = nptemperature[nptemperature.Year .== baselineyear, "T"][1] .+ deviationtokelvin
u₀ = [m₀, T₀];
p₀ = [150., 20.5];
Tcalibrationtime = (2020., 2100.);
npprob = ODEProblem(Fnp!, u₀, Tcalibrationtime, p₀); solve(npprob, AutoVern9(Rosenbrock23()))

tdxs = Tcalibrationtime[1] .≤ nptemperature.Year .≤ Tcalibrationtime[2];
matchyears = nptemperature.Year[tdxs];
matchtemperature = nptemperature.T[tdxs] .+ deviationtokelvin;
function matchtemperatureloss(p)
    sol = solve(npprob, AutoVern9(Rosenbrock23()); p = p, saveat = matchyears)

    return mean(abs2, last.(sol.u) .- matchtemperature)
end

tmatchf = Optimization.OptimizationFunction((p, _) -> matchtemperatureloss(p), Optimization.AutoForwardDiff())
tmatchprob = Optimization.OptimizationProblem(tmatchf, p₀)
result = Optimization.solve(tmatchprob, PolyOpt())

begin # Check calibration
    p̂ = result.u
    ts = co2calibrationtime
    npprob = ODEProblem(Fnp!, u₀, extrema(ts), p̂)
    
    sol = solve(npprob, AutoVern9(Rosenbrock23()))
end

begin # Plot temperature calibration
    T̂ = sol(ts).u
    
    tdx = first(ts) .≤ nptemperature.Year .≤ last(ts)

    fig = vspan(collect(Tcalibrationtime); alpha = 0.4, c = :lightgray, label = "Calibration period", xlabel = L"Year, $t$", ylabel = raw"Temperature [°C]", xlims = extrema(ts), legend = :topleft, xticks = 2020:20:2150)

    plot!(fig, nptemperature[tdx, "Year"], nptemperature[tdx, "T"]; label = "FaIRv2 SSP5-8.5", c = :black, linewidth = 2.5, alpha = 0.5)
    plot!(fig, nptemperature[tdx, "Year"], nptemperature[tdx, "T lower"]; fillrange = nptemperature[tdx, "T upper"], c = :black, linewidth = 0., alpha = 0.1)

    plot!(fig, ts, getindex.(T̂, 2) .- deviationtokelvin; label = "Median fit", c =:black, linestyle = :dash, linewidth = 2.5)

    if SAVEFIG
        savefig(fig, joinpath(PLOTPATH, "temperaturecalibration.tikz"))
    end

    fig
end

let # Compute error
    error = @. (getindex(T̂, 2) - deviationtokelvin) - nptemperature[tdx, "T"]
    indx = first(Tcalibrationtime) .≤ nptemperature.Year[tdx] .≤ last(Tcalibrationtime)
    insampleerror = maximum(abs, error[indx])
    outerror = maximum(abs, error[.!indx])

    @printf "Maximum in-sample error: %.5f °C, out-of-sample error: %.5f °C" insampleerror outerror
end

begin # Hogg calibration
    todaydx = findfirst(co2equivalence.Year .== baselineyear)
    frac = co2equivalence[todaydx, "CO2 Concentration"] ./ co2equivalence[todaydx, "Concentration"]
    N₀ = 286.65543 / frac[1]

    G₀, G₁ = result.u

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
albedo = Albedo(Tᶜ = 1.5)
tspan = (2020., 2150.); # Time span in years

du = similar(u₀); # Allocate memory for the derivative
system!(du, u₀, albedo, 2020.); # Call the system to ensure it works
prob = ODEProblem(system!, u₀, tspan, albedo);
probsol = solve(prob, AutoVern9(Rosenbrock23()))

mediandf = modeldfs[3];
row = mediandf[mediandf.Year .== 2100, :];
const targetΔT = row.T_Coupled[1] - row.T_Uncoupled[1]

function terminalΔTloss(sim)::Float64
    t = last(tspan)
    _, T, Tˡ = sim(t)
    ΔT = T - Tˡ

    return abs2(targetΔT - ΔT)
end

lossbyparam = @closure Δλ -> begin
    albedo = Albedo(Tᶜ = 1.5, Δλ = Δλ)
    sol = solve(prob, AutoVern9(Rosenbrock23()); p = albedo)

    return terminalΔTloss(sol)
end

_, Δλ = gssmin(lossbyparam, 0., 0.1)
plot(0:0.001:0.05, lossbyparam; ylabel = "Loss", xlabel = L"\Delta\lambda", linewidth = 2, c = :black, xlims = (0, 0.05))

sol = solve(prob, AutoVern9(Rosenbrock23()); p = Albedo(Tᶜ = 2.0, Δλ = Δλ))

begin
    solfig = plot(xlabel = "Year", ylabel = "Temperature [°C]", title = "Temperature Dynamics with Tipping Element", xticks = (0:10:80, 2020:10:2100))

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

    nullclinefig = plot(xlabel = "CO2 concentration [p.p.m.]", xticks = (mticks, mlabels), ylabel = "Temperature [°C]", yticks = (Tticks, Tlabels))

    plot!(nullclinefig, last.(mstable), Tspace; label = "With tipping element", color = :darkred, linewidth = 2.5, linestyle = :dash)
    plot!(nullclinefig, first.(mstable), Tspace; label = "Without tipping element", color = :darkblue, linewidth = 2.5, linestyle = :dash)

end

# Save outcome of calibration in file
jldopen(calibrationpath, "w+") do file
    @pack! file = hogg, calibration, albedo
end