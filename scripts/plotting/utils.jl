using Model: AbstractModel, JumpModel, TippingModel
using Printf: format, Format

const kelvintocelsius = 273.15

function stringifydeviation(ΔT; digits = 2)
    fsign = ΔT > 0 ? "+" : ""
    fmt = Format("$fsign%0.$(digits)f°")
    return format(fmt, ΔT)
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

function labelofmodel(model)
    if model isa LinearModel
        "No tipping element"
    elseif model isa TippingModel
        L"T^c = %$(model.feedback.Tᶜ - model.hogg.Tᵖ)"
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
