using Revise

using DotEnv, UnPack, DataStructures
using CSV, JLD2
using DataFrames, TidierData
using SparseArrays, LinearAlgebra, LinearAlgebra

filepath = "data/AR6_Scenarios_Database_World_v1.1.csv"; @assert isfile(filepath)
rawdf = CSV.read(filepath, DataFrame)
scenarios = @chain rawdf begin
    @filter(Region == "World")
    @filter(Variable ∈ ("Price|Carbon", "Emissions|CO2", "GDP|PPP"))
    @filter(contains(Model, "IMAGE 3.0"))
    @filter(contains(Scenario, "SSP"))
    @group_by(Model, Scenario)
end

npscenario = "SSP5-Baseline"
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

dfnp = dfs[npscenario]
filter!(((scenario, df), ) -> scenario != npscenario, dfs)
nmodels = length(dfs)

Xs = Matrix{Float64}[]
ys = Vector{Float64}[]
for (k, (scenario, df)) in enumerate(dfs)
    idx = findall(mac -> mac > -10., df[:, "Price|Carbon"])
    if isempty(idx) continue end

    mac = df[idx, "Price|Carbon"]
    Y = df[idx, "GDP|PPP"]
    E = df[idx, "Emissions|CO2"] / 1000

    jdx = findall(y -> y ∈  df[idx, "Year"], dfnp.Year)
    Eⁿᵖ = dfnp[jdx, "Emissions|CO2"] / 1000
    Yⁿᵖ = dfnp[jdx, "GDP|PPP"]

    β′ = @. max(mac * Eⁿᵖ / Y, 1e-3)
    ε = @. max(1 - E / Eⁿᵖ, 1e-3)

    Δt = df[idx, :Year] .- 2012.
    X = [ones(length(Δt)) Δt]
    y = @. log(β′) - log(ε)

    push!(Xs, X); push!(ys, y)
end

# Create model fixed effects
modelidxs = Int[]
for (k, X) in enumerate(Xs)
    append!(modelidxs, fill(k, size(X, 1)))
end

# Create fixed effects matrix
nobs = sum(size(X, 1) for X in Xs)
FE = sparse(1:nobs, modelidxs, 1.0, nobs, nmodels)
X = sparse(hcat(FE, vcat(Xs...)))
y = vcat(ys...)

coeff = (X'X) \ (X'y)
α₀, α₁ = coeff[(end - 1):end]

# Compute standard errors using pseudo-inverse for singular matrices
residuals = y - X * coeff
σ² = sum(residuals.^2) / (length(y) - rank(Matrix(X)))  # Use rank instead of size
XtX_pinv = pinv(Matrix(X'X))  # Pseudo-inverse for singular matrix
V = σ² * XtX_pinv

# Extract variances for α₀ and α₁ (only diagonal elements are reliable with pseudo-inverse)
var_α₀ = V[end-1, end-1]
var_α₁ = V[end, end]

# Standard errors of original coefficients
se_α₀ = sqrt(max(var_α₀, 0))  # Ensure non-negative
se_α₁ = sqrt(max(var_α₁, 0))

# Transformed coefficients
ω₀ = log(α₀)
ωᵣ = -α₁

# Standard errors of transformed coefficients using delta method
# For ω₀ = log(α₀): SE(ω₀) = SE(α₀) / α₀
se_ω₀ = se_α₀ / abs(α₀)
se_ωᵣ = se_α₁

println("Original coefficients:")
println("α₀ = $(round(α₀, digits=6)) ± $(round(se_α₀, digits=6))")
println("α₁ = $(round(α₁, digits=6)) ± $(round(se_α₁, digits=6))")
println("\nTransformed coefficients:")
println("ω₀ = log(α₀) = $(round(ω₀, digits=6)) ± $(round(se_ω₀, digits=6))")
println("ωᵣ = -α₁ = $(round(ωᵣ, digits=6)) ± $(round(se_ωᵣ, digits=6))")