using JSON
include("../../scripts/utils/saving.jl")

inputdir = isempty(ARGS) ? "data/simulation-medium" : first(ARGS);
inputfiles = listfiles(inputdir; exclude = ["terminal", "jump"]); # FIXME: Jump is not working for now.

obj = Dict("inputfiles" => inputfiles);

open("jobs/experiments/parameters.json", "w") do file
    JSON.print(file, obj, 2)
end