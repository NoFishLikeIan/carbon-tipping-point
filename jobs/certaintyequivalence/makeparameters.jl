using JSON

rras = [2.0, 10.0];
eiss = [0.75, 1.5];
remote = [2.5, 3.5];

tippingparameters = Dict{String, Union{Float64, Bool}}[];

for rra in rras, eis in eiss, remotethreshold in remote
    push!(tippingparameters, Dict("rra" => rra, "eis" => eis, "remotethreshold" => remotethreshold))
end

obj = Dict("certaintyequivalence" => tippingparameters)

open("jobs/certaintyequivalence/parameters.json", "w") do f
    JSON.print(f, obj, 2)
end