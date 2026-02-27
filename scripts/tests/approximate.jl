using Revise, BenchmarkTools
using Model, Grid
using FastChebInterp, Interpolations
using StaticArrays
using UnPack, DataStructures

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
simpath = "data/simulation";
paths = rand(loadsimulationpaths(simpath; exclude = ["terminal", "linear"]), 10) |> sort
values = OrderedDict(k => loadtotal(p) for (k, p) in paths)

order = (20, 20, 10, 5)
H = chebyshevrepresentation(values, order);