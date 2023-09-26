using BenchmarkTools

using UnPack  
using StatsBase
using NLsolve, Optim
using LinearAlgebra
using Base.Iterators: product

include("../src/model/init.jl")
include("../src/utils/derivatives.jl")

# -- Test over a cube
const dims = 3; # Dimension state space excluding time
const Z₀ = reshape([hogg.T₀, log(hogg.M₀), log(economy.Y₀)], (dims, 1)) |> tounit;
const grid = paddedrange(3ϵ, 1f0 - 3ϵ; pad = 2ϵ);
const n = length(grid)^dims;

begin
    const Z = Matrix{Float32}(undef, dims, n)
    const U₀ = Matrix{Float32}(undef, 1, n)

    for (i, zᵢ) ∈ product(fill(grid, dims)...) |> enumerate
        Z[:, i] .= zᵢ
        U₀[i] = (zᵢ[3]^2 - zᵢ[1]^2 + zᵢ[2]^2) / 4f0 - 0.7f0
    end
end;

const V₀ = U₀ .* economy.V̲
const X = fromunit(Z);
const ∇U₀ = central∇V(U₀); @time central∇V(U₀);
const ∇V₀ = expand∂(∇U₀);

const α₀ = abatement(economy.t₁, X, ∇V₀); @time abatement(economy.t₁, X, ∇V₀);
const χ₀ = consumption(economy.t₁, V₀, ∇V₀); @time consumption(economy.t₁, V₀, ∇V₀);
const w = drift(economy.t₁, X, α₀, χ₀); @time drift(economy.t₁, X, α₀, χ₀)

# TODO
Vₜ = copy(V₀);

∇(V::Matrix{Float32}) = (V ./ economy.V̲) |> central∇V |> expand∂


G(Vₜ::Matrix{Float32}, t::Float32, X::Matrix{Float32}) = G!(copy(Vₜ), t, X)
function G!(Vₜ::Matrix{Float32}, t::Float32, X::Matrix{Float32})
    ∇Vₜ = ∇(Vₜ)
    α = abatement(t, X, ∇Vₜ)
    χ = consumption(economy.t₁, Vₜ, ∇Vₜ)
    w = drift(t, X, α, χ)

    Vₜ .= Vₜ .+ ϵ .* f.(χ, X[[3], :] , V, Ref(economy))

    for j in axes(Vₜ, 2)
        Vₜ[:, j] .= Vₜ[:, j] .+ ϵ .* ∇Vₜ[:, j]'w[:, j]
    end

    Vₜ .= Vₜ .+ ∂²T(Vₜ) * (hogg.σ²ₜ / 2f0)
    
    return Vₜ
end

Vₜ₋₁ = copy(Vₜ);
@time G!(Vₜ₋₁, economy.t₁, X);