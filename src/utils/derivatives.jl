ϵ = cbrt(eps(Float32))
ϵ⁻¹ = inv(ϵ)

function ∂(f, x)
    second = (f(x .- 2ϵ) - f(x .+ 2ϵ)) ./ 12  
    first = 2(f(x .+ ϵ) - f(x .- ϵ)) ./ 3

    return (second .+ first) .* (ϵ⁻¹)
end

function ∂(f, x, v)
    second = (f(x .- 2ϵ * v) - f(x .+ 2ϵ * v)) ./ 12
    first =  2(f(x .+ ϵ * v) - f(x .- ϵ * v)) ./ 3

    return (second .+ first) .* (ϵ⁻¹)
end

function basis(n)
    Id = Matrix{Float32}(I(n))

    return eachcol(Id)
end

∇(h, x) = reduce(vcat, ∂(h, x, e) for e in basis(size(x, 1)))