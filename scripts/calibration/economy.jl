using Revise

using DotEnv, UnPack, DataStructures
using CSV, JLD2
using DataFrames, TidierData
using SparseArrays, LinearAlgebra, StaticArrays
using Optim
using Plots, LaTeXStrings

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

        append!(ts, t)
        append!(εs, abated)
        append!(β′s, mac)
    end
end

function Model.β′(t, ε, p::StaticVector{4, S}) where S <: Real
    ω̄, Δω, ρ, b = p
    return β′(t, ε, Abatement(ω̄, Δω, ρ, b))
end

function objective(p, params)
    ts, εs, β′s = params
    residual = 0.

    for k in eachindex(β′s)
        mac = β′(ts[k], εs[k], p)
        residual += abs2(β′s[k] - mac)
    end

    return residual / length(β′s)
end

# Initial parameter guess
p₀ = MVector(0.00043, 0.05506, 0.0148, 1.95) # DICE initial condition
params = (ts, εs, β′s);

# Constrain parameters to reasonable ranges
lower = MVector(0., 0., 0., 1.)
upper = MVector(1., 1., Inf, 2.8)  # Increased upper bound for b

result = optimize(p -> objective(p, params), lower, upper, p₀, Fminbox(LBFGS()))

begin
    ω̄, Δω, ρ, b = round.(result.minimizer, digits = 6)

    println("Calibrated parameters:")
    println("ω̄ = $(round(ω̄, digits=6))")
    println("Δω = $(round(Δω, digits=6))")  
    println("ρ = $(round(ρ, digits=6))")
    println("b = $(round(b, digits=4))")
    println("Objective value: $(round(result.minimum, digits=4))")
end

abatement = Abatement(ω̄, Δω, ρ, b)
begin # Calculate R² for goodness of fit 
    predicted = [β′(t, ε, abatement) for (t, ε) in zip(ts, εs)]
    residual = sum(abs2, β′s .- predicted)
    total = sum(abs2, β′s .- mean(β′s))
    R² = 1 - residual / total
    println("R² = $(round(R², digits=4))")
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