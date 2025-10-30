using Revise

using DotEnv, UnPack, DataStructures
using CSV, JLD2
using DataFrames, TidierData
using SparseArrays, LinearAlgebra, StaticArrays
using Optim
using Plots, LaTeXStrings, Printf
using FastClosures
using DifferentialEquations, StaticArrays

using Model

include("constants.jl")

DATAPATH = "data"
filepath = joinpath(DATAPATH, "AR6_Scenarios_Database_World_v1.1.csv"); @assert isfile(filepath)

calibrationpath = joinpath(DATAPATH, "calibration")
if !isdir(calibrationpath) mkpath(calibrationpath) end

investments = Investment{Float64}()
begin # Load scenarios data
    rawdf = CSV.read(filepath, DataFrame)
    scenarios = @chain rawdf begin
        @filter(Region == "World")
        @filter(Model ∈ ("REMIND-MAgPIE 2.1-4.2",))
        # Scenarios filters
        @filter(!occursin(r"EN_NPi.*", Scenario))
        @filter(!occursin("CEMICS_SSP2-Npi", Scenario))
        @filter(!occursin(r"Delayed Transition", Scenario))
        @filter(Scenario == "EN_NoPolicy" || 
                occursin(r"PkBudg\d+", Scenario) ||          # Carbon budget scenarios
                occursin(r"EN_INDCi.*_\d+f?$", Scenario) ||  # Carbon budget scenarios  
                occursin(r"NGFS2_.*(2°C|Net-Zero)", Scenario) ||  # NGFS climate scenarios
                occursin(r"CEMICS_SSP\d-1p5C", Scenario) ||  # 1.5°C scenarios
                occursin(r"CEMICS_SSP\d-2C", Scenario))      # 2°C scenarios
        @group_by(Model, Scenario)
    end
end

begin # Construct scenarios dataframes
    npscenario = "EN_NoPolicy"  # True no-policy baseline from REMIND-MAgPIE
    parsefloat(year) = parse(Float64, year)
    dfs = Dict{String, DataFrame}()
    for (k, scenario) in pairs(scenarios)
        df = @chain DataFrame(scenario) begin
            @select(!(:Model, :Scenario, :Region, :Unit))
            stack(Not(:Variable), variable_name="Year", value_name="Value")
            unstack(:Variable, :Value)
            dropmissing!()
            @mutate(Year = parsefloat(Year))
            @filter(Year ≥ 2020.)
        end

        isnpscenario = k.Scenario == npscenario
        hasvalidprices = ("Price|Carbon" ∈ names(df) && !all((df[!, "Price|Carbon"]) .≈ 0))

        if isnpscenario || hasvalidprices
            dfs[k.Scenario] = df
        end
    end
    dfnp = dfs[npscenario]
    filter!(((scenario, df), ) -> scenario != npscenario, dfs)
    nmodels = length(dfs)
end


begin # Fill in data coefficients
    macupper = 0.5

    emissionkey = "AR6 climate diagnostics|Infilled|Emissions|Kyoto Gases (AR6-GWP100)"

    ts = Float64[]
    εs = Float64[]
    β′s = Float64[]
    for (k, (scenario, df)) in enumerate(dfs)
        carbonprice = df[:, "Price|Carbon"] # $ / tCO2
        Y = df[:, "GDP|PPP"]
        years = df.Year
        dfnpmatch = filter(r -> r.Year in years, dfnp)
        
        E = df[:, emissionkey] ./ 1000
        Eⁿᵖ = dfnpmatch[:, emissionkey] ./ 1000

        mac = @. carbonprice * Eⁿᵖ / Y
        abated = @. 1 - E / Eⁿᵖ
        t = df[:, :Year] .- 2020.

        macdx = mac .< macupper
        append!(ts, t[macdx])
        append!(εs, abated[macdx])
        append!(β′s, mac[macdx])
    end
end

begin
    @printf "Data diagnostics:\n"
    @printf "N observations: %d\n" length(β′s)
    @printf "MAC range: %.4f to %.4f\n" minimum(β′s) maximum(β′s)
    @printf "MAC mean: %.4f, median: %.4f\n" mean(β′s) median(β′s)
    @printf "Abatement range: %.4f to %.4f\n" minimum(εs) maximum(εs)
    @printf "Time range: %.1f to %.1f\n" minimum(ts) maximum(ts)
end

function Model.β′(t, ε, p::StaticVector{4, S}) where S <: Real
    ω̄, Δω, ρ, bᵢ = p
    return β′(t, ε, Abatement(ω̄, Δω, ρ, bᵢ))
end

function huber(p, params; δ = 1e-2)
    ts, εs, β′s = params
    residual = 0.0
    
    for k in eachindex(β′s)
        predicted = β′(ts[k], εs[k], p)
        observed = β′s[k]
        
        residual += δ^2 * (√(1 + abs2((predicted - observed) / δ)) - 1)
    end
    
    return residual
end

# First estimate abatement ε ≤ 1
begin
    p₀ = MVector(1e-7, 0.09, 0.025, 2.8)
    params = (ts, εs, β′s);
    
    lb = MVector(0., 0., 0., 0.)
    ub = MVector(Inf, Inf, Inf, Inf)

    obj = @closure p -> huber(p, params; δ = 0.05)
    result = optimize(obj, lb, ub, p₀, Fminbox(LBFGS()))

    ω̄, Δω, ρ, bᵢ = round.(result.minimizer, digits = 6)
    abatement = Abatement(ω̄, Δω, ρ, bᵢ)
    hambelabatement = Abatement(p₀...)

    @printf "\nCalibrated abatement parameters:\nω̄ = %.6f\nΔω = %.6f\nρ = %.6f\nb = %.4f\nObjective value: %.4f\n" ω̄ Δω ρ bᵢ result.minimum

    result
end

begin # Create a plot of β′ for various values of t
    εspace = 0:0.01:maximum(εs)

    timespace = sort(unique(ts))
    cs = Dict(timespace .=> palette(:viridis, timespace))
    tspace = timespace[1:2:end]

    fig = plot(xlabel=L"Abatement rate $\varepsilon$", ylabel=L"Marginal abatement cost $\beta\prime$", title="Marginal Abatement Cost Function")

    for t in tspace
        plot!(fig, εspace, ε -> β′(t, ε, abatement), label="$(2020 + t)", linewidth=3, c = cs[t])
    end

    # Add scatter plot of actual data points
    zcolors = [cs[t] for t in ts]
    scatter!(fig, εs, β′s; markersize = 3, alpha = 0.5, c = zcolors, label = false, markerstrokewidth = 0.)

    fig
end

# Save outcome of calibration in file
jldopen(joinpath(calibrationpath, "abatement.jld2"), "w+") do file
    @pack! file = abatement
end