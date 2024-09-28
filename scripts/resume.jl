using Pkg
Pkg.resolve(); Pkg.instantiate();

using Dates: now
using Base.Threads: nthreads
using UnPack: @unpack

include("arguments.jl") # Import argument parser
parsedargs = ArgParse.parse_args(resumeargtable)
@unpack filepath, verbose = parsedargs

if !isfile(filepath) throw("File not found: $filepath") end

if (verbose ≥ 1)
    println("$(now()): ", "Resuming $filepath with $(nthreads()) threads.")    
    flush(stdout)
end

include("markov/backward.jl")
include("utils/saving.jl")
timesteps, values, policies, G, model = loadtotal(filepath);

Fₜ₊ₕ = values[:, :, 1];
Fₜ = similar(Fₜ₊ₕ); F = (Fₜ, Fₜ₊ₕ);
policy = policies[:, :, :, 1];

τ = model.economy.τ - minimum(timesteps); # Last simulation step
queue = DiagonalRedBlackQueue(G; initialvector = τ * ones(prod(size(G))));

backwardsimulation!(queue, F, policy, model, G; verbose = verbose, cachepath = filepath, overwrite = false)