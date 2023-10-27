using UnPack, JLD2, DotEnv
using DifferentialEquations

using Model, Utils

using Polyester

include("../src/timederivative.jl")

# -- Generate state cube

const economy = Model.Economy();
const hogg = Model.Hogg();
const albedo = Model.Albedo();

const instance = (economy, hogg, albedo);

const n = 51
const statedomain = [
    (hogg.T₀, hogg.T̄, n + 50), 
    (log(hogg.M₀), log(hogg.M̄), n), 
    (log(economy.Y̲), log(economy.Ȳ), n)
];

const Ω = Utils.makegrid(statedomain);
const X = Utils.fromgridtoarray(Ω);

# -- Terminal condition
DATAPATH = get(DotEnv.config(), "DATAPATH", "data/") 
@load joinpath(DATAPATH, "terminal.jld2") V̄

W₀ = similar(X, size(X)[1:3]);
for mdx in axes(W₀, 2)
    W₀[:, mdx, :] .= copy(V̄);
end

# -- Backward simulation
function backwardstep!(∂Wₜ, Wₜ, params, t)
    W₀, tmp_cache = params

    G!(∂Wₜ, get_tmp(tmp_cache, Wₜ), t, X, W₀ .- Wₜ, Ω, instance, economy)
end


const tmpcache = DiffCache(Array{Float32}(undef, length.(Ω)..., 4));

params = (W₀, )
prob = ODEProblem(backwardstep!, W₀, (0f0, 80f0),)