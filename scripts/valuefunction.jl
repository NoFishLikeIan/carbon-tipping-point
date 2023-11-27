using UnPack, JLD2, DotEnv
using DifferentialEquations
using PreallocationTools
using ImageFiltering
using Polyester

using Model, Utils
include("../src/timederivative.jl")

using BenchmarkTools
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

W₀data = similar(X, size(X)[1:3]);
for mdx in axes(W₀data, 2)
    W₀data[:, mdx, :] .= copy(V̄);
end

W₀ = BorderArray(W₀data, Utils.paddims(W₀data, 2));

# -- Backward simulation
function backwardstep!(∂Wₜ, Wₜ, params, t)
    W₀, ∇Vcache, ∂²Vcache, policycache, wcache = params

    G!(∂Wₜ, 
        get_tmp(∇Vcache, Wₜ), get_tmp(∂²Vcache, Wₜ), get_tmp(policycache, Wₜ), get_tmp(wcache, Wₜ),
        t, X, W₀ .- Wₜ, Ω, instance, calibration
    )
end



const policyinner = ones(Float32, gridsize..., 2) ./ 2f0;
const policy = BorderArray(policyinner, Utils.paddims(policyinner, 1, (1, 2, 3)));

const ∇Vcache = DiffCache(Array{Float32}(undef, gridsize..., 4));
const ∂²Vcache = DiffCache(Array{Float32}(undef, gridsize));
const policycache = DiffCache(policy);
const wcache = DiffCache(Array{Float32}(undef, gridsize..., 3));

params = (W₀, ∇Vcache, ∂²Vcache, policycache, wcache);
∂ₜW₀ = similar(W₀.inner); t = 0f0;

println("Running tests...")
G!(∂ₜW₀, 
    get_tmp(∇Vcache, ∂ₜW₀), 
    get_tmp(∂²Vcache, ∂ₜW₀), 
    get_tmp(policycache, W₀), 
    get_tmp(wcache, ∂ₜW₀),
    t, X, W₀, Ω, instance, calibration
)
@btime backwardstep!($∂ₜW₀, $W₀, $params, $t);

# prob = ODEProblem(backwardstep!, W₀, (0f0, 80f0), params);

# integrator = init(prob, Tsit5());