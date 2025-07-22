using Revise

using DotEnv, UnPack
using CSV, DataFrames, JLD2

using DifferentialEquations, DifferentialEquations.EnsembleAnalysis
using DiffEqParamEstim, Optimization, OptimizationOptimJL
using Roots, FastClosures

using Plots, LaTeXStrings, Printf
default(label = false, dpi = 180)
pgfplotsx()

using Model, Grid

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

env = DotEnv.config()
datapath = get(env, "DATAPATH", "data/")
coupleddir = joinpath(datapath, "deutloff", "model_output", "coupled_ensemble"); @assert isdir(coupleddir)
uncoupleddir = joinpath(datapath, "deutloff", "model_output", "uncoupled_ensemble"); @assert isdir(uncoupleddir)

probabilities = groupby(
    loadhierdataframe(:Probability => joinpath(coupleddir, "tip_prob_total.csv"); groupkeys = [:TE]),
    [:Scenario, :TE]
)

begin # Temperature dataframe
    temperaturedf = loadhierdataframe(:T_Coupled => joinpath(coupleddir, "T.csv"); groupkeys = [:Quantile]);
    temperaturedf.T_Uncoupled = loadhierdataframe(:T_Uncoupled => joinpath(uncoupleddir, "T.csv"); groupkeys = [:Quantile]).T_Uncoupled

    temperature = groupby(temperaturedf, [:Scenario, :Quantile])
end;

# Emissions dataframe
begin
    concentrationdf = loadhierdataframe("Concentration" => joinpath(coupleddir, "C.csv"); groupkeys = [:Particle, :Quantile])
    concentrationdf.Quantile = safeparse.(Float64, concentrationdf.Quantile)

    concentration = groupby(concentrationdf, [:Scenario, :Particle, :Quantile])
end;

begin # Figure CO₂ concentration
    qs = [0.05, 0.5, 0.95];
    figscenarios = ["SSP1 1.9", "SSP2 4.5", "SSP4 3.4", "SSP5 8.5"]
    cmap = palette(:viridis, length(figscenarios); rev = true)

    co2fig = plot(xlabel = "Year", yaxis = "Concentration (ppm)", title = "Carbon Dioxide Concentration", xlims = (2000, 2150))
    excesstfig = plot(xlabel = "Year", yaxis = "Temperature [°]", title = "Additional temperature", xlims = (2012, 2150))
    
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

# Doubling CO2 year
const ppmpreindustrial = 280.0 # ppm

function ecsyear(concentration, scenario, quantile)
    index = (scenario, "carbon_dioxide", quantile)
    df = concentration[index]
    jdx = findfirst(m -> m ≥ 3ppmpreindustrial, df.Concentration)

    return isnothing(jdx) ? df.Year[end] : df.Year[jdx]
end

function getadditionalecs(quantile, temperature, year, scenario)
    df = temperature[(scenario, quantile)]
    row = df[df.Year .== year, :]
    Tuncoupled, Tcoupled = row.T_Uncoupled[1], row.T_Coupled[1]

    return Tcoupled - Tuncoupled
end

begin
    co2fig = plot(xlabel = "Year", yaxis = "Concentration (ppm)", title = "Carbon Dioxide Concentration", xlims = (2000, 2150))
    excesstfig = plot(xlabel = "Year", yaxis = "Temperature [°]", title = "Temperature Deviations", xlims = (2000, 2150))
    
    let scenario = "SSP5 8.5", c = :darkred
        co2 = Tuple(concentration[(scenario, "carbon_dioxide", q)] for q in qs);
        temp = Tuple(temperature[(scenario, q)] for q in qs);
        
        plot!(co2fig, co2[1].Year, co2[1].Concentration; color = c, fillrange = co2[3].Concentration, fillalpha = 0.05, linewidth = 0., label = false)
        plot!(co2fig, co2[2].Year, co2[2].Concentration; label = scenario, color = c, linewidth = 2.5)

        doubleyears = [ecsyear(concentration, scenario, q) for q in qs];
        hline!(co2fig, [3ppmpreindustrial], linestyle = :dash, color = :black, linewidth = 1.5)
        scatter!(co2fig, doubleyears, fill(3ppmpreindustrial, length(doubleyears)); color = :black)

        plot!(excesstfig, temp[1].Year, temp[1].T_Coupled - temp[1].T_Uncoupled; color = c, fillrange = temp[3].T_Coupled - temp[3].T_Uncoupled, fillalpha = 0.05, linewidth = 0., label = false)
        plot!(excesstfig, temp[2].Year, temp[2].T_Coupled - temp[2].T_Uncoupled; label = scenario, color = c, linewidth = 2.5, ylims = (0, Inf))

        for (i, year) in enumerate(doubleyears)
            additionalecs = [getadditionalecs(q, temperature, year, scenario) for q in qs]
            scatter!(excesstfig, fill(year, length(additionalecs)), additionalecs; color = :black, markersize = 5)
            plot!(excesstfig, [year, year], [0, maximum(additionalecs)]; linestyle = :dash, color = :black, markersize = 5)
        end
    end

    impulsefig = plot(co2fig, excesstfig; layout = (1, 2), size = 600 .* (2√2, 1), margins = 10Plots.mm, legend = :topleft)

    # savefig(impulsefig, "plots/impulse.png")

    impulsefig
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

const modeldfs = constructquantiledf(concentration, temperature, "SSP5 8.5"; qs = [0.05, 0.5, 0.95])

# Dynamics of the calibration
const calibration = load_object("data/calibration.jld2")
const hogg = Hogg()

function system!(du, u, Δλ, t)
    m, T, Tˡ = @view u[1:3]

    # Temperature with tipping element
    ΔT = 1.8
    λ₁ = 0.31
    Tᶜ = 2.2 # [K] pre-industrial temperature
    T₁ = Tᶜ + hogg.Tᵖ
    T₂ = T₁ + ΔT
    inflexion = (T₁ + T₂) / 2

    Lₜ = Model.sigmoid(T - inflexion) 
    λₜ = λ₁ - Δλ * Lₜ
    r = hogg.S₀ * (1 - λₜ) - hogg.η * T^4

    # Temperature with no tipping element
    rˡ = Model.radiativeforcing(Tˡ, hogg)

    # GHG forcing
    g = Model.ghgforcing(m, hogg)
    du[1] = γ(t, calibration)
    du[2] = (g + r) / hogg.ϵ
    du[3] = (g + rˡ) / hogg.ϵ
end
function noise!(Σ, u, parameters, t)
	Σ[1] = hogg.σₘ
    Σ[2] = hogg.σₜ / hogg.ϵ
    Σ[3] = hogg.σₜ / hogg.ϵ
end

u₀ = [log(hogg.M₀), hogg.T₀, hogg.T₀]; # Initial conditions
Δλ₀ = 0.03; # Initial parameters for albedo
tspan = (2020., 2100.) .- 2020.; # Time span in years

du = similar(u₀); # Allocate memory for the derivative
system!(du, u₀, Δλ₀, 0.0); # Call the system to ensure it works

prob = SDEProblem(system!, noise!, u₀, tspan, Δλ₀);
probsol = solve(prob)

ensemble = EnsembleProblem(prob);
sim = solve(ensemble, SRIW1(); saveat = tspan[end], trajectories = 10_000);

hasfailed(sim) = any((!SciMLBase.successful_retcode(s.retcode) for s in sim))

const kelvintocelsius = 273.15
mediandf = modeldfs[2];
row = df[df.Year .== round(Int64, tspan[2] + 2020), :];
const targetΔT = row.T_Coupled[1] - row.T_Uncoupled[1]

function loss(sim)    
    totloss = 0.0

    if hasfailed(sim)
        totloss = Inf
    else
        t = last(tspan)
        year = round(Int64, t + 2020)
        _, T, Tˡ = timepoint_median(sim, t)
        ΔT = T - Tˡ

        totloss += abs2(targetΔT - ΔT)
    end

    return totloss
end

lossbyparam = @closure Δλ -> begin
    prob = SDEProblem(system!, noise!, u₀, tspan, Δλ)
    ensemble = EnsembleProblem(prob)
    sim = solve(ensemble, SRIW1(); saveat = tspan[2], trajectories = 50_000)

    return loss(sim)
end

_, Δλ = gssmin(lossbyparam, 0., 10.)

prob = SDEProblem(system!, noise!, u₀, tspan, Δλ)
sol = solve(prob)
plot(sol; idxs = [2, 3], label = ["T" "Tˡ"], xlabel = "Year", ylabel = "Temperature [°C]", title = "Temperature Dynamics with Tipping Element", size = (800, 400), legend = :topright, xticks = (0:10:80, 2020:10:2100))