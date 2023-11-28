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