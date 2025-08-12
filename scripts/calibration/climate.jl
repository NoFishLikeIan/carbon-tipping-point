using Revise

using DotEnv, UnPack
using CSV, DataFrames, JLD2
using StaticArrays
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis
using Optimization, OptimizationOptimJL, OptimizationPolyalgorithms
using ForwardDiff, DifferentiationInterface

using Roots, FastClosures
using Statistics, LinearAlgebra
using Interpolations
using LogExpFunctions

# Plotting setup
using Plots, Printf, PGFPlotsX, Colors, ColorSchemes, LaTeXStrings

pgfplotsx()
default(label = false, dpi = 180, linewidth = 2.5)
push!(PGFPlotsX.CUSTOM_PREAMBLE, 
    raw"\usepgfplotslibrary{fillbetween}",
    raw"\usepackage{siunitx}",
    raw"\DeclareSIUnit{\ppm}{p.p.m.}"
);

using Model, Grid

includet("../utils/simulating.jl")

PLOTPATH = "papers/job-market-paper/submission/plots"
DATAPATH = "data"
SAVEFIG = true
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

if isinteractive() # Figure CO₂ concentration
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

if isinteractive() # Figure CO₂ concentration in the no policy scenario
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

if isinteractive() # Figure fraction of forcing
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
end;

function concentrationloss(p, optparams)
    ts, γ̂ = optparams

    loss = zero(eltype(γ̂))
    for (i, t) in enumerate(ts)
        loss += abs2(γ(t, Tuple(p)) - γ̂[i])
    end

    return loss
end

begin # Initialize the CO₂e calibration problem
    p₀ = MVector(
        0.005,   # p[1] - t² coefficient in numerator
        0.05,    # p[2] - t coefficient in numerator
        0.006,   # p[3] - constant term in numerator (≈ starting value)
        0.1,     # p[4] - t² coefficient in denominator
        1.0,     # p[5] - t coefficient in denominator
        1.0,     # p[6] - constant term in denominator
        0.0001,  # p[7] - offset amplitude
        0.005    # p[8] - offset decay rate
    )

    ts = range(0, τ - 1.; step = 1.)
    γ̂ = diff(m)
    optparams = (ts, γ̂)

    γfn = Optimization.OptimizationFunction(concentrationloss, AutoForwardDiff())
    γprob = Optimization.OptimizationProblem(γfn, p₀, optparams)

    result = solve(γprob, PolyOpt(); g_tol = 1e-10)
end

# --- Implied emissions
begin
    m₀ = first(m)
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

if isinteractive() # Check simulated fit
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
    η = Hogg{Float64}().η # Stefan-Boltzmann constant in Wm⁻²K⁻⁴
    S₀ = Hogg{Float64}().S₀ # Incoming radiative forcing
    
    ghgcalibrationhorizon = 80.
    ghgspan = baselineyear .+ (0, ghgcalibrationhorizon)
    t₀, t₁ = ghgspan

    m̂ = log.(co2calibrationdf[t₀ .≤ co2calibrationdf.Year .≤ t₁, "Concentration"] ./ Mᵖ)
    T̂ = nptemperature[t₀ .≤ nptemperature.Year .≤ t₁, "T"] .+ deviationtokelvin # Temperature in Kelvin

    X = hcat(ones(length(m̂)), m̂)
    y = η * T̂.^4 .- S₀
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
    defaults = (calibration, Float64(baselineyear), η, S₀, G₀, G₁)
    
    T₀ = 1.4 + deviationtokelvin
    u₀ = SVector(m₀, T₀)
    ϵ₀ = 0.15
    linearprob = ODEProblem(Flinear, u₀, ghgspan, (ϵ₀, defaults))
    testsol = solve(linearprob, AutoVern9(Rodas5P()); saveat = t₀:t₁) # Ensure the problem is well-posed
    
    tdx = t₀ .≤ nptemperature.Year .≤ t₁;
    matchtemperature = nptemperature[tdx, "T"]

    optparams = (linearprob, defaults, matchtemperature);

    lossfn = @closure ϵ -> temperatureloss(ϵ, optparams)

    _, ϵ = gssmin(lossfn, 0., 100.)
end

if isinteractive() # Check calibration
    u₀ = SVector(m₀, T₀)
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

begin # Hogg definition
    todaydx = findfirst(co2equivalence.Year .== baselineyear)
    N₀ = co2equivalence[todaydx, "Concentration"] * 286.65543 / co2equivalence[todaydx, "CO2 Concentration"]

    hogg = Hogg(
        T₀ = T₀, M₀ = M₀, Mᵖ = Mᵖ, N₀ = N₀,
        ϵ = ϵ, G₀ = G₀, G₁ = G₁, σₜ = ϵ / 10
    )
end

# TIPPING POINT CALIBRATION
if isinteractive()
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

# Dynamics of the calibration
function coupledsystem(u, parameters, t)
    hogg, calibration, Tᶜ, ΔS, L = parameters
    feedback = Feedback(promote(Tᶜ, ΔS, L)...)

    m, Tˡ, T = u

    dm = γ(t - calibration.baselineyear, calibration)
    dTˡ = μ(Tˡ, m, hogg) / hogg.ϵ
    dT = μ(T, m, hogg, feedback) / hogg.ϵ

    return SVector(dm, dTˡ, dT)
end

tedfs = constructquantiledf(concentration, temperature, npscenario; qs = [0.05, 0.5, 0.95])

function tippingelementloss(p, optparameters)
    coupledprob, ΔTtrajectory = optparameters
    Tᶜ, ΔS, L = p
    hogg, calibration = coupledprob.p[1:2]

    parameters = (hogg, calibration, Tᶜ, ΔS, L)
    sol = solve(coupledprob, RadauIIA5(); p = parameters, saveat = 1., verbose = false, reltol = 1e-8)

    if SciMLBase.successful_retcode(sol)
        ΔT = getindex.(sol.u, 3) .- getindex.(sol.u, 2)
        
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
    
    res[1] = (L * ΔS * 2 / 3) - 16hogg.η * (Tᶜ)^3 # > 0 Fold constraint

    # > 0 Parameter constraints
    res[2] = Tᶜ - hogg.T₀
    res[3] = L
    res[4] = ΔS

    return res
end

begin # Calibrate adjustment speed
    u₀ = SVector(m₀, T₀, T₀)
    
    tecalibrationhorizon = 100. + baselineyear
    centurydx = findfirst(==(tecalibrationhorizon), co2calibrationdf.Year)
    mtarget = log(co2calibrationdf[centurydx, "Concentration"] / Mᵖ)

    Tᶜ₀ = 4. + hogg.T₀; L₀ = 7.; ΔS₀ = 10.
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
        optparameters = (coupledprob, ΔTtrajectory)

        objfunction = Optimization.OptimizationFunction(tippingelementloss, adtype; cons = constraints!)
        optproblem = Optimization.OptimizationProblem(objfunction, p₀, optparameters; lcons, ucons)

        teresult = solve(optproblem, IPNewton(); iterations = 10_000)
        
        if !SciMLBase.successful_retcode(teresult)
            @warn @sprintf "Result %i not converged.\n" i
        else
            @printf "Problem %i converged with ΔTᶜ=%.3f, ΔS=%.3f, L=%.3f, error=%.3e.\n" i (teresult.u[1] - hogg.Tᵖ) teresult.u[2] teresult.u[3] teresult.objective
            
            res = constraints(teresult.u, optparameters)
    
            if !all(lcons .< res .< ucons)
                @warn @sprintf "Problem %i: constraints not satisfied. Residuals: %.3f, %.3f, %.3f, %.3f" i res[1] res[2] res[3] res[4]
            end
        end

        Tᶜ, ΔS, L = teresult.u
        feedback = Feedback(Tᶜ, ΔS, L)

        push!(feedbacks, feedback)
    end
end

function extendedcoupledsystem!(du, u, parameters, t)
    hogg, calibration, feedbacks = parameters
    m = u[1]

    du[1] = γ(t - calibration.baselineyear, calibration)
    du[2] = μ(u[2], m, hogg)

    for (i, feedback) in enumerate(feedbacks)
        du[i + 2] = μ(u[i + 2], m, hogg, feedback)
    end
end;

if isinteractive()
    u₀ = [m₀, T₀, T₀, T₀, T₀]
    parameters = (hogg, calibration, feedbacks)

    extenedcoupledprob = ODEProblem(extendedcoupledsystem!, u₀, (baselineyear, tecalibrationhorizon), parameters)

    sol = solve(extenedcoupledprob, RadauIIA5(); saveat = 1., reltol = 1e-8)

    solfig = plot(xlabel = "Year", ylabel = L"Temperature $[\si{\degree}]$", title = "Temperature Dynamics with Tipping Element", legendtitle = "Quantile")

    plot!(solfig, sol.t, getindex.(sol.u, 2); c = :black, label = "Linear")
    
    qs = (0.05, 0.5, 0.95)
    c = palette(:reds, 3)
    for i in 1:3
        plot!(solfig, sol.t, getindex.(sol.u, i + 2); labels = qs[i], c = c[i])
    end
    
    solfig
end

# Check Nullclines
if isinteractive()
    Tspace = range(hogg.T₀, hogg.T₀ + 6.; length = 101)
    nullclinefig = plot(ylabel = "Temperature [°]", xlabel = "CO2e concentration (ppm)")
    
    for (i, feedback) in enumerate(feedbacks)
        mnullcline = [mstable(T, hogg, feedback) for T in Tspace]
        plot!(nullclinefig, mnullcline, Tspace; label = qs[i], c = c[i])
    end

    nullclinefig
end

# Save outcome of calibration in file
jldopen(calibrationpath, "w+") do file
    feedbacklower, feedback, feedbackhigher = feedbacks
    @pack! file = hogg, calibration, feedbacklower, feedback, feedbackhigher
end