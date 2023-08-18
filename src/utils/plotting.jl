kelvintocelsius = 273.15
xpreindustrial = 14 + kelvintocelsius

function stringtempdev(x::Real; digits = 2)
    fsign = x > 0 ? "+" : ""
    fmt = Printf.Format("$fsign%0.$(digits)f")
    return Printf.format(fmt, x)
end

function makedevxlabels(from, to, climate; step = 0.5, withcurrent = false, digits = 2)

    preindustrialx = range(from, to; step = step)
    xticks = preindustrialx .+ xpreindustrial

    xlabels = [stringtempdev(x, digits = digits) for x in preindustrialx]

    if !withcurrent
        return (xticks, xlabels)
    end

    xlabels = [xlabels..., "\$x_0\$"]
    xticks = [xticks..., first(climate).x₀]
    idxs = sortperm(xticks)
    
    return (xticks[idxs], xlabels[idxs])
end

function generateframes(total, frames)
	step = total ÷ frames
	return [range(1, total - 2step; step = step)..., total]
end