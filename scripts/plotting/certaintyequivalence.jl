using JLD2, UnPack

begin # Load
    cepath = "data/certaintyequivalence"
    results = Dict{String, Float64}[]

    for filename in readdir(cepath)
        filepath = joinpath(cepath, filename)
        result = JLD2.load_object(filepath)
        push!(results, result)
    end
end


function costofregret(result)
    (result["cē"] - result["ceᵖ"]) - (result["ce̲"] - result["ceʷ"])
end

costofregret.(results)