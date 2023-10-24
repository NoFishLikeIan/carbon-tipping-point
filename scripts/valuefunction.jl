using Base.Threads
using UnPack

include("../src/utils/grids.jl")
include("../src/utils/derivatives.jl")
include("../src/model/initialisation.jl")

# -- Generate state cube
const n = 51
const statedomain::Vector{Domain} = [
    (hogg.T₀, hogg.T̄, n), 
    (log(hogg.M₀), log(hogg.M̄), n), 
    (log(economy.Y̲), log(economy.Ȳ), n)
];
const Ω = makeregulargrid(statedomain);
const X = fromgridtoarray(Ω);

# -- Generate action square
const m = 51
const actiondomain::Vector{Domain} = [
    (1f-3, 1f0 - 1f-3, 11), (1f-3, 1f0 - 1f-3, 11)
]

const Γ = makeregulargrid(actiondomain);
const P = fromgridtoarray(Γ);

# -- Terminal condition