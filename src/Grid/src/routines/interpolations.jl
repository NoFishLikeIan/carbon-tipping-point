function inverselinearinterpolation(xs, ys)
    qs = collect(range(0, 1; length = size(xs, 1)))
    inverselinearinterpolation!(qs, xs, ys)
end

function inverselinearinterpolation!(qs, xs, ys)
    @inbounds for (i, q) in enumerate(qs)
        # Find the interval containing q
        idx = searchsortedfirst(ys, q)
        
        if idx == 1
            qs[i] = xs[1]
        elseif idx > length(ys)
            qs[i] = xs[end]
        else
            y1, y2 = ys[idx-1], ys[idx]
            x1, x2 = xs[idx-1], xs[idx]
            t = (q - y1) / (y2 - y1)
            qs[i] = x1 + t * (x2 - x1)
        end
    end

    return qs
end