"""
Solves the HJB equation for a two regime system with critical value x̂. It assumes that the lowregime[end] and highregime[1] are equidistant from x̂.
"""
function solvepiecewisevalue(hbj::Function, grid::Tuple)

    lowregime, highregime = grid
    x̂ = (highregime[1] + lowregime[end]) / 2

    lowsol, lowresidual = pdesolve(
        hbj, 
        OrderedDict(:x => lowregime), 
        OrderedDict(:v => -100. * ones(length(lowregime)));
        verbose = false
    )

    if lowresidual > 1e-3
        @warn "Low regime residual is too large: $lowresidual"
    end

    highsol, highresidual = pdesolve(
        hbj, 
        OrderedDict(:x => highregime), 
        OrderedDict(:v => -100. * ones(length(highregime)));
        verbose = false
    )

    if highresidual > 1e-3
        @warn "High regime residual is too large: $highresidual"
    end

    lowspline = Spline1D(lowregime, lowsol[:v])
    highspline = Spline1D(highregime, highsol[:v])

    """
    Compute the ν-th derivative of the value function at x. If ν = 0, computes v(x).
    """
    function v(x; ν::Int64 = 0)
        spl = x ≤ x̂ ? lowspline : highspline

        return ν > 0 ? derivative(spl, x; nu = ν) : spl(x)
    end

    return v, lowresidual + highresidual
end