function plotvectorfield(xs, ys, g::Function; plotkwargs...)
    fig = plot()
    plotvectorfield!(fig, xs, ys, g; plotkwargs...)
    return fig
end

function plotvectorfield!(figure, xs, ys, g::Function; rescale=1, plotkwargs...)

    xlims = extrema(xs)
    ylims = extrema(ys)

    N, M = length(xs), length(ys)
    xm = repeat(xs, outer=M)
    ym = repeat(ys, inner=N)

    field = g.(xm, ym)

    scale = rescale * (xlims[2] - xlims[1]) / min(N, M)
    u = @. scale * first(field)
    v = @. scale * last(field)

    steadystates = @. (u ≈ 0) * (v ≈ 0)

    u[steadystates] .= NaN
    v[steadystates] .= NaN

    z = (x -> √(x'x)).(field)

    quiver!(
        figure, xm, ym;
        quiver=(u, v), line_z=repeat(z, inner=4),
        aspect_ratio=1, xlims=xlims, ylims=ylims,
        c=:batlow, colorbar=false,
        plotkwargs...
    )

end

