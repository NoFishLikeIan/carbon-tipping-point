using JLD2, UnPack

begin # Load
    cepath = "data/certaintyequivalence"
    results = Dict{String, Float64}[]

    filelist = readdir(cepath)

    for filename in filelist
        filepath = joinpath(cepath, filename)
        result = JLD2.load_object(filepath)
        push!(results, result)
    end
end


function costofregret(result)
    (result["cē"] - result["ceᵖ"]) - (result["ce̲"] - result["ceʷ"])
end

cor = costofregret.(results)

println(Dict(filelist .=> cor))
