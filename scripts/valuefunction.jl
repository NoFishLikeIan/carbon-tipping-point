using UnPack, JLD2, DotEnv
using DifferentialEquations
using PreallocationTools

using BenchmarkTools

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

const gridsize = Tuple(last.(statedomain))
const Ω = Utils.makegrid(statedomain);
const X = Utils.fromgridtoarray(Ω);

# -- Terminal condition and calibration
DATAPATH = get(DotEnv.config(), "DATAPATH", "data/") 
V̄ = load(joinpath(DATAPATH, "terminal.jld2"), "V̄");
const calibration = load(joinpath(DATAPATH, "calibration.jld2"), "calibration");

W₀ = similar(X, size(X)[1:3]);
for mdx in axes(W₀, 2)
    W₀[:, mdx, :] .= copy(V̄);
end

# -- Backward simulation
function backwardstep!(∂Wₜ, Wₜ, params, t)
    W₀, ∇Vcache, ∂²Vcache, policy, w = params

    G!(∂Wₜ, 
        get_tmp(∇Vcache, Wₜ), get_tmp(∂²Vcache, Wₜ), get_tmp(policy, Wₜ), get_tmp(w, Wₜ),
        t, X, W₀ .- Wₜ, Ω, instance, calibration
    )
end

const ∇Vcache = DiffCache(Array{Float32}(undef, gridsize..., 4));
const ∂²Vcache = DiffCache(Array{Float32}(undef, gridsize));
const policycache = DiffCache(Array{Float32}(undef, gridsize..., 2));
const wcache = DiffCache(Array{Float32}(undef, gridsize..., 3));

params = (W₀, ∇Vcache, ∂²Vcache, policycache, wcache);
∂ₜW₀ = similar(W₀); t = 0f0;

println("Running tests...")
G(t, X, W₀, Ω, instance, calibration);
@btime G($t, $X, $W₀, $Ω, $instance, $calibration);
@btime backwardstep!($∂ₜW₀, $W₀, $params, $t);

# prob = ODEProblem(backwardstep!, W₀, (0f0, 80f0), params);

# integrator = init(prob, Tsit5());