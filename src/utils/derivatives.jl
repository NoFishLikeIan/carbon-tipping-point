const ϵ = cbrt(eps(Float32))
const ϵ⁻¹ = inv(ϵ)
const ϵ⁻² = inv(ϵ^2)

function paddedrange(from::Float32, to::Float32)
    collect(range(from - 2ϵ, to + 2ϵ; step = ϵ)[:, :]')
end

const S₁¹ = reshape(ϵ⁻¹ .* [-1f0, 8f0, 0f0, -8f0, 1f0] ./ 12f0, 1, 5, 1, 1)
const S₁² = reshape(ϵ⁻² .* [-1f0, 16f0, -30f0, 16f0, -1f0] ./ 12f0, 1, 5, 1, 1)

function ∂(y)
    reshape(
        NNlib.conv(reshape(y, 1, size(y, 2), 1, 1), S₁¹),
        1, (size(y, 2) - 4)
    )
end

function ∂²(y)
    reshape(
        NNlib.conv(reshape(y, 1, size(y, 2), 1, 1), S₁²),
        1, (size(y, 2) - 4)
    )
end