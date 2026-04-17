const kelvintocelsius = 273.15

const PGF_FIGURE_SIZES = Dict{Symbol, Tuple{Float64, Float64}}(
    :panel => (0.34, 0.24),
    :panel_pair => (0.36, 0.26),
    :panel_wide => (0.48, 0.34),
    :panel_bar => (0.50, 0.36),
    :compact => (0.50, 0.32),
    :medium => (0.60, 0.43),
    :wide => (0.72, 0.44),
    :wide_tall => (0.72, 0.50),
    :square => (0.60, 0.60),
    :policy => (0.80, 0.50),
    :full => (0.90, 0.68),
    :full_alt => (0.78, 0.60),
)

function pgf_figsize(kind::Symbol; basis = "\\textwidth")
    width, height = PGF_FIGURE_SIZES[kind]
    return @pgf { width = string(width, basis), height = string(height, basis) }
end

function stringifydeviation(ΔT; digits = 2)
    fsign = ΔT > 0 ? "+" : ""
    fmt = Printf.Format("$fsign%0.$(digits)f°")
    return Printf.format(fmt, ΔT)
end
function makedeviationtickz(from, to; step = 0.5, digits = 2, addedlabels = Tuple{String, Float64}[])

    ticks = collect(range(from, to; step = step))
    labels = [stringifydeviation(x; digits = digits) for x in ticks]

    if isempty(addedlabels)
        return ticks, labels
    end

    ticks = [ticks..., last.(addedlabels)...]
    labels = [labels..., first.(addedlabels)...]
    idxs = sortperm(ticks)
    
    return ticks[idxs], labels[idxs]
end

function labelsofclimate(climate::C) where {C <: Climate}
    if C <: Model.PiecewiseLinearClimate
        "No tipping element"
    elseif C <: TippingClimate
        Tᶜ = round(climate.feedback.Tᶜ; digits = 2)
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

function abatementcolorbar(Ē)
    if Ē ≤ 1.0
        cgrad([RGB(0.8392, 0.1882, 0.1529), :white])
    else
        cgrad(:RdBu, [0., 1 / Ē, 1])
    end
end