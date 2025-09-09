const kelvintocelsius = 273.15

function stringifydeviation(ΔT; digits = 2)
    fsign = ΔT > 0 ? "+" : ""
    fmt = Printf.Format("$fsign%0.$(digits)f°")
    return Printf.format(fmt, ΔT)
end
function makedeviationtickz(from, to, hogg; step = 0.5, digits = 2, addedlabels = Tuple{String, Float64}[])

    preindustrialdev = range(from, to; step = step)
    ticks = hogg.Tᵖ .+ preindustrialdev

    labels = [stringifydeviation(x; digits = digits) for x in preindustrialdev]

    if isempty(addedlabels)
        return (ticks, labels)
    end

    ticks = [ticks..., last.(addedlabels)...]
    labels = [labels..., first.(addedlabels)...]
    idxs = sortperm(ticks)
    
    return (ticks[idxs], labels[idxs])
end

function labelsofclimate(climate::C) where {C <: Climate}
    if C <: Model.PiecewiseLinearClimate
        "No tipping element"
    elseif C <: TippingClimate
        Tᶜ = round(model.feedback.Tᶜ - model.climate.hogg.Tᵖ; digits = 2)
        L"T^c = %$(Tᶜ)"
    else
        error("Model type not implemented")
    end
end

function smoother(vs, n)
    out = similar(vs)
    m = length(vs)

    for i in axes(vs, 1)
        l = max(i - n, 1)
        r = min(i + n, m)

        out[i] = mean(@view vs[l:r])
    end

    return out
end
