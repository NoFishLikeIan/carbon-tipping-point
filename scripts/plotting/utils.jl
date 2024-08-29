using Printf: format, Format

function stringifydeviation(ΔT; digits = 2)
    fsign = ΔT > 0 ? "+" : ""
    fmt = Format("$fsign%0.$(digits)f°")
    return format(fmt, ΔT)
end


function makedeviationtickz(from, to, model; step = 0.5, withcurrent = false, digits = 2)

    preindustrialdev = range(from, to; step = step)
    ticks = model.hogg.Tᵖ .+ preindustrialdev

    labels = [stringifydeviation(x; digits = digits) for x in preindustrialdev]

    if !withcurrent
        return (ticks, labels)
    end

    labels = [labels..., "\$x_0\$"]
    ticks = [ticks..., first(climate).x₀]
    idxs = sortperm(ticks)
    
    return (ticks[idxs], labels[idxs])
end