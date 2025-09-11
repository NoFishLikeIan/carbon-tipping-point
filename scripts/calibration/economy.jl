using Revise

using DotEnv, UnPack, DataStructures
using CSV, JLD2
using DataFrames, TidierData
using SparseArrays, LinearAlgebra, StaticArrays
using Optim
using Plots, LaTeXStrings, Printf

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
    ts = Float64[]
    εs = Float64[]
    β′s = Float64[]

    for (k, (scenario, df)) in enumerate(dfs)
        carbonprice = df[:, "Price|Carbon"]
        Y = df[:, "GDP|PPP"]
        E = df[:, "Emissions|CO2"] / 1000

        # Filter out observations with zero or very low carbon prices
        validprices = carbonprice .> 0.
        
        if !any(validprices)
            continue 
        end

        jdx = findall(y -> y ∈  df[validprices, "Year"], dfnp.Year)
        Eⁿᵖ = dfnp[jdx, "Emissions|CO2"] / 1000
        Yⁿᵖ = dfnp[jdx, "GDP|PPP"]

        mac = @. max(carbonprice[validprices] * Eⁿᵖ / Y[validprices], 1e-6)
        abated = @. max(1 - E[validprices] / Eⁿᵖ, 1e-6)
        t = df[validprices, :Year] .- 2020.
        
        valid_idx = mac .< 2.
        
        if any(valid_idx)
            append!(ts, t[valid_idx])
            append!(εs, abated[valid_idx])
            append!(β′s, mac[valid_idx])
        end
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

function objective(p, params)
    ts, εs, β′s = params
    residual = 0.

    for k in eachindex(β′s)
        predicted = β′(ts[k], εs[k], p)
        observed = β′s[k]
        
        residual += abs2(observed - predicted)
    end

    return residual / length(β′s)
end

# Initial parameter guess with lower b
p₀ = MVector(0.00043, 0.05506, 0.0148, 2.2) # Lower initial b
params = (ts, εs, β′s);

# Constrain parameters to reasonable ranges with tighter bound on b
lower = MVector(0., 0., 0., 2.)
upper = MVector(Inf, Inf, Inf, 2.8)  # Much tighter upper bound for b

result = optimize(p -> objective(p, params), lower, upper, p₀, Fminbox(LBFGS()))

begin
    ω̄, Δω, ρ, b = round.(result.minimizer, digits = 6)

    @printf "Calibrated parameters:\n"
    @printf "ω̄ = %.6f\n" ω̄
    @printf "Δω = %.6f\n" Δω
    @printf "ρ = %.6f\n" ρ
    @printf "b = %.4f\n" b
    @printf "Objective value: %.4f\n" result.minimum
end

abatement = Abatement(ω̄, Δω, ρ, b)
begin 
    predicted = [β′(t, ε, abatement) for (t, ε) in zip(ts, εs)]
    residual = sum(abs2, β′s .- predicted)
    total = sum(abs2, β′s .- mean(β′s))
    R² = 1 - residual / total
    @printf "R² = %.4f\n" R²
end

begin # Create a plot of β′ for various values of t
    tspace = 0:20:80
    εspace = 0:0.01:1.4

    timespace = sort(unique(ts))
    cs = Dict(timespace .=> palette(:viridis, timespace))

    fig = plot(xlabel=L"Abatement rate $\varepsilon$", ylabel=L"Marginal abatement cost $\beta\prime$", title="Marginal Abatement Cost Function")

    # Add scatter plot of actual data points
    zcolors = [cs[t] for t in ts]
    scatter!(fig, εs, β′s; markersize = 2, alpha=0.3, c = zcolors, label = false)

    for t in tspace
        plot!(fig, εspace, ε -> β′(t, ε, abatement), label="$(2020 + t)", linewidth=3, c = cs[t])
    end

    fig
end

# Save outcome of calibration in file
jldopen(joinpath(calibrationpath, "abatement.jld2"), "w+") do file
    @pack! file = abatement
end