using Revise

using DotEnv, UnPack, DataStructures
using CSV, JLD2
using DataFrames, TidierData
using SparseArrays, LinearAlgebra, StaticArrays
using Optim
using Plots, LaTeXStrings, Printf
using FastClosures

using Model

DATAPATH = "data"
filepath = joinpath(DATAPATH, "AR6_Scenarios_Database_World_v1.1.csv"); @assert isfile(filepath)

calibrationpath = joinpath(DATAPATH, "calibration")
if !isdir(calibrationpath) mkpath(calibrationpath) end

begin # Load scenarios data
    rawdf = CSV.read(filepath, DataFrame)
    scenarios = @chain rawdf begin
        @filter(Region == "World")
        @filter(Variable ∈ ("Price|Carbon", "Emissions|CO2", "GDP|PPP"))
        @filter(Model ∈ ("REMIND-MAgPIE 2.1-4.2",))
        @filter(!occursin(r"EN_NPi.*", Scenario))
        @filter(!occursin("CEMICS_SSP2-Npi", Scenario))
        @filter(!occursin(r"Delayed Transition", Scenario))
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

        isnp = k.Scenario == npscenario

        if !isnp && ("Price|Carbon" ∉ names(df) || all((df[!, "Price|Carbon"]) .≈ 0))
            continue
        end

        dfs[k.Scenario] = df
    end
end

dfnp = dfs[npscenario]
filter!(((scenario, df), ) -> scenario != npscenario, dfs)
nmodels = length(dfs)

begin # Fill in data coefficients
    macmax = Inf

    ts = Float64[]
    εs = Float64[]
    β′s = Float64[]
    for (k, (scenario, df)) in enumerate(dfs)
        carbonprice = df[:, "Price|Carbon"]
        Y = df[:, "GDP|PPP"]
        E = df[:, "Emissions|CO2"] / 1000

        # Filter out observations with zero or very low carbon prices
        validprices = carbonprice .> 0.
        tdx = dfnp.Year .> 2030

        if !any(tdx .& validprices)
            continue
        end

        Eⁿᵖ = dfnp[tdx .& validprices, "Emissions|CO2"] / 1000
        Yⁿᵖ = dfnp[tdx .& validprices, "GDP|PPP"]

        mac = @. carbonprice[tdx .& validprices] * Eⁿᵖ / Y[tdx .& validprices]
        abated = @. 1 - E[tdx .& validprices] / Eⁿᵖ
        t = df[tdx .& validprices, :Year] .- 2020.

        macdx = mac .< macmax
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
    ω̄, Δω, ρ, b = p
    return β′(t, ε, Abatement(ω̄, Δω, ρ, b))
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
    p₀ = MVector(0.00264, 0.09, 0.025, 2.8)
    params = (ts, εs, β′s);

    lower = MVector(0., 0., 0., 0.)
    upper = MVector(Inf, Inf, Inf, Inf)

    obj = @closure p -> huber(p, params)
    result = optimize(obj, lower, upper, p₀, Fminbox(LBFGS()))

    ω̄, Δω, ρ, b = round.(result.minimizer, digits = 6)
    abatement = Abatement(ω̄, Δω, ρ, b)

    @printf "\nCalibrated abatement parameters:\nω̄ = %.6f\nΔω = %.6f\nρ = %.6f\nb = %.4f\nObjective value: %.4f\n" ω̄ Δω ρ b result.minimum
end

begin # Create a plot of β′ for various values of t
    εspace = 0:0.01:maximum(εs)

    timespace = sort(unique(ts))
    cs = Dict(timespace .=> palette(:viridis, timespace))
    tspace = timespace[2:2:end]

    fig = plot(xlabel=L"Abatement rate $\varepsilon$", ylabel=L"Marginal abatement cost $\beta\prime$", title="Marginal Abatement Cost Function")

    for t in tspace
        plot!(fig, εspace, ε -> β′(t, ε, abatement), label="$(2020 + t)", linewidth=3, c = cs[t])
    end

    # Add scatter plot of actual data points
    zcolors = [cs[t] for t in ts]
    scatter!(fig, εs, β′s; markersize = 2, alpha=0.3, c = zcolors, label = false)

    fig
end



# Save outcome of calibration in file
jldopen(joinpath(calibrationpath, "abatement.jld2"), "w+") do file
    @pack! file = abatement
end