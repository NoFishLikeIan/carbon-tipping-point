using Revise
using Test: @test
using UnPack: @unpack
using BenchmarkTools
using JLD2

includet("../utils/saving.jl")
includet("../markov/terminal.jl")
includet("../markov/backward.jl")

filepath = "data/simulation-local/constrained/albedo/growth/Tc=1,50_ρ=0,01500_θ=10,00_ψ=0,75_σT=1,5840_σm=0,0078_ωᵣ=0,01756_ξ=0,000075_υ=3,250.jld2";
@assert isfile(filepath)

timesteps, values, policies, G, model = loadtotal(filepath);

begin
    totalsize = prod(size(G))

    queue = DiagonalRedBlackQueue(G; initialvector = (model.economy.τ - minimum(timesteps)) * ones(totalsize))

	Δts = zeros(size(G, 1) * size(G, 2))
	cluster = first(dequeue!(queue))
end;

begin
    Fₜ₊ₕ = values[:, :, 1]
    Fₜ = similar(Fₜ₊ₕ); F = (Fₜ, Fₜ₊ₕ);

    policy = policies[:, :, :, 1]
end

backwardstep!(Δts, F, policy, cluster, model, G);