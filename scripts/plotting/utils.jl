using Printf: format, Format

const kelvintocelsius = 273.15

function stringifydeviation(ΔT; digits = 2)
    fsign = ΔT > 0 ? "+" : ""
    fmt = Format("$fsign%0.$(digits)f°")
    return format(fmt, ΔT)
end

function makedeviationtickz(from, to, model; step = 0.5, digits = 2, addedlabels = NTuple{String, Float64}[])

    preindustrialdev = range(from, to; step = step)
    ticks = model.hogg.Tᵖ .+ preindustrialdev

    labels = [stringifydeviation(x; digits = digits) for x in preindustrialdev]

    if isempty(addedlabels)
        return (ticks, labels)
    end

    ticks = [ticks..., last.(addedlabels)...]
    labels = [labels..., first.(addedlabels)...]
    idxs = sortperm(ticks)
    
    return (ticks[idxs], labels[idxs])
end