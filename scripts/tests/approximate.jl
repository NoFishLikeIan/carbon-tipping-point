using Revise
using Model, Grid
using FastChebInterp, Interpolations
using StaticArrays
using UnPack

using DataStructures, JLD2

includet("../../src/valuefunction.jl")
includet("../../src/extend/valuefunction.jl")
includet("../../src/extend/model.jl")
includet("../../src/extend/grid.jl")

# IO 
includet("../utils/loading.jl")
includet("../utils/saving.jl")

includet("../utils/approximate.jl")


## Load data
simpath = "data/simulation-dense";
paths = loadsimulationpaths(simpath);

values, model, G = loadtotal(paths[2.0]);
H = chebyshevrepresentation(values, G; order = (100, 100, 20));