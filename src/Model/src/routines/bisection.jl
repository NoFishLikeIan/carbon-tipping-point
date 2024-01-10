@inline bisect(a, b) = a/2 + b/2

function bisection(f, a_, b_; atol = 2eps(Float64), increasing = sign(f(b_)))
    a_, b_ = minmax(a_, b_)
    c = middle(a_, b_)
    z = f(c) * increasing
    if z > 0 #
        b = c
        a = typeof(b)(a_)
    else
        a = c
        b = typeof(a)(b_)
    end
    while abs(a - b) > atol
        c = middle(a, b)
        if f(c) * increasing > 0 #
            b = c
        else
            a = c
        end
    end
    a, b
end

const ϕ⁻¹ = (√(5.) - 1.) / 2.
const ϕ⁻² = (3. - √(5.)) / 2.

function gss(f, a, b; tol = 1e-5)
    Δ = b - a
    
    n = ceil(Int, log(tol / Δ) / log(ϕ⁻¹))
    
    c = a + Δ * ϕ⁻²
    d = a + Δ * ϕ⁻¹

    yc = f(c)
    yd = f(d)

    for _ in 1:n
        if yc > yd
            b = d
            d, yd = c, yc
            Δ *= ϕ⁻¹
            c = a + ϕ⁻² * Δ
            yc = f(c)
        else
            a = c
            c, yc = d, yd
            Δ *= ϕ⁻¹
            d = a + Δ * ϕ⁻¹
            yd = f(d)
        end
    end

    if yc > yd
        return yc, (a + d) / 2
    else
        return yd, (c + b) / 2
    end
end