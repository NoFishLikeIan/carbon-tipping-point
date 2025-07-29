const secondstoyears = 60 * 60 * 24 * 365.25
const Gtonoverppm = 1 / 7.821


@Base.kwdef struct Albedo
	Tᶜ::Float64  # Initiation of Albedo from pre industrial levels
    ΔT::Float64 = 0.18 # [K] Speed of transition
    λ₁::Float64 = 0.31 # Initial radiation reflection coefficient
    Δλ::Float64 = 0.015454304088339633 # Loss in reflection coefficient
    β::Float64 = 8.24 # Adjustment speed of reflection loss
end

function updateTᶜ(Tᶜ, albedo::Albedo)
    Albedo(Tᶜ = Tᶜ, ΔT = albedo.ΔT, λ₁ = albedo.λ₁, Δλ = albedo.Δλ, β = albedo.β)
end

Base.@kwdef struct Jump
    j₂::Float64 = -0.0029
    j₁::Float64 = 0.0568
    j₀::Float64 = -0.0577

    i₀::Float64 = -0.25
    i₁::Float64 = 0.95
    
    e₁::Float64 = 2.8
    e₂::Float64 = -0.3325
end

Base.@kwdef struct Hogg
    # Current and pre-industrial data temperature and carbon concentration
    T₀::Float64 = 288.22214675250292 # [K]
    Tᵖ::Float64 = 287.15 # [K]
    M₀::Float64 = 558.748949198944 # [p.p.m. CO₂-eq]
    Mᵖ::Float64 = 383.149374377142 # [p.p.m. CO₂-eq]

    N₀::Float64 = 286.65543 # [p.p.m.]
    
    σₜ::Float64 = 1.584 # Standard deviation of temperature
    σₘ::Float64 = 0.0078 # Standard deviation of CO₂ 

    # Climate sensitwivity
    S₀::Float64 = 340.5 # [W / m²] Mean solar radiation

    ϵ::Float64 = 5e8 / secondstoyears # years * [J / m² K] / s Heat capacity of the ocean
    η::Float64 = 5.67e-8 # Stefan-Boltzmann constant 
    
    G₁::Float64 = 20.5 # [W / m²] Effect of CO₂ on radiation budget
    G₀::Float64 = 150 # [W / m²] Pre-industrial GHG radiation budget
    
    # Decay rate of carbon concentration
    aδ::Float64 = 0.0176
    bδ::Float64 = -27.63
    cδ::Float64 = 384.8
end

Base.broadcastable(m::Albedo) = Ref(m)
Base.broadcastable(m::Jump) = Ref(m)
Base.broadcastable(m::Hogg) = Ref(m)

"Obtain an Hogg calibration consistent with the Albedo calibration"
function equilibriumHogg(albedo::Albedo; b = (330., 380.))::Hogg
    d = Hogg()
    eq = @closure S₀ ->  Model.Mstable(d.T₀, Hogg(S₀ = S₀), albedo) - d.M₀

    S₀ = find_zero(eq, b)

    return Hogg(S₀ = S₀)
end

"Decay of carbon"
function δₘ(M, hogg::Hogg)
    N = M * (hogg.N₀ / hogg.M₀)
    return hogg.aδ * exp(-(N - hogg.cδ)^2 / hogg.bδ^2)
end

function δₘ⁻¹(δ, hogg::Hogg)
    N = hogg.cδ + hogg.bδ * √(log(hogg.aδ / δ))
    return log(N * hogg.M₀ / hogg.N₀)
end

# Albedo functions
sigmoid(x, β) = inv(1 + exp(-x * β))
sigmoid′(x, β) = β * sigmoid(x, β) * (1 - sigmoid(x, β))

function L(T, hogg::Hogg, albedo::Albedo)
    T₁ = albedo.Tᶜ + hogg.Tᵖ
    T₂ = T₁ + albedo.ΔT
    inflexion = (T₁ + T₂) / 2

    sigmoid(T - inflexion, albedo.β)
end
function L′(T, hogg::Hogg, albedo::Albedo)
    T₁ = albedo.Tᶜ + hogg.Tᵖ
    T₂ = T₁ + albedo.ΔT
    inflexion = (T₁ + T₂) / 2

    sigmoid′(T - inflexion, albedo.β)
end

λ(T, hogg::Hogg, albedo::Albedo) = albedo.λ₁ - albedo.Δλ * L(T, hogg, albedo)
λ′(T, hogg::Hogg, albedo::Albedo) = -albedo.Δλ * L′(T, hogg, albedo)

"Radiative forcing."
radiativeforcing(T, hogg::Hogg, albedo::Albedo) = hogg.S₀ * (1 - λ(T, hogg, albedo)) - hogg.η * T^4
radiativeforcing(T, hogg::Hogg) = hogg.S₀ * 0.69 - hogg.η * T^4
radiativeforcing′(T, hogg::Hogg, albedo::Albedo) = -hogg.S₀ * λ′(T, hogg, albedo) - 4hogg.η * T^3

"Greenhouse gases"
ghgforcing(m, hogg::Hogg) = hogg.G₀ + hogg.G₁ * m
ghgforcing⁻¹(r, hogg::Hogg) = (r - hogg.G₀) / hogg.G₁


"Drift temperature dynamics"
μ(T, m, hogg::Hogg, albedo::Albedo) = radiativeforcing(T, hogg, albedo) + ghgforcing(m, hogg)
μ(T, m, hogg::Hogg) = radiativeforcing(T, hogg) + ghgforcing(m, hogg)

"Compute CO₂ concentration consistent with temperature T"
mstable(T, hogg::Hogg, albedo::Albedo) = ghgforcing⁻¹(-radiativeforcing(T, hogg, albedo), hogg)
mstable(T, hogg::Hogg) = ghgforcing⁻¹(-radiativeforcing(T, hogg), hogg)

function Tstable(m, hogg::Hogg, albedo::Albedo)
    find_zeros(T -> mstable(T, hogg, albedo) - m, hogg.Tᵖ, 1.2hogg.Tᵖ)
end
function Tstable(m, hogg::Hogg)
    find_zeros(T -> mstable(T, hogg) - m, hogg.Tᵖ, 1.2hogg.Tᵖ)
end

"Compute equilibrium climate sensitivity"
function ecs(hogg::Hogg)
    Tstable(log(2), hogg) .- hogg.Tᵖ
end
function ecs(hogg::Hogg, albedo::Albedo)
    Tstable(log(2), hogg, albedo) .- hogg.Tᵖ
end

function potential(T, m, hogg::Hogg, albedo::Albedo)
	@unpack λ₁, Δλ = albedo
    T₁ = albedo.Tᶜ + hogg.Tᵖ
    T₂ = T₁ + albedo.ΔT
    inflexion = (T₁ + T₂) / 2
	G = ghgforcing(m, hogg)

	(hogg.η / 5) * T^5 - G * T - (1 - λ₁) * hogg.S₀ * T - hogg.S₀ * Δλ * log(1 + exp(T - inflexion))
end

function density(T, m, hogg::Hogg, albedo::Albedo; normalisation = 1e-5)
    exp(-normalisation * potential(T, m, hogg, albedo))
end

"Size of jump"
function increase(T, hogg::Hogg, jump::Jump)
    ΔT = T - hogg.Tᵖ

    jump.j₀ + jump.j₁ * ΔT + jump.j₂ * ΔT^2
end


"Arrival rate of jump"
function intensity(T, hogg::Hogg, jump::Jump)
    ΔT = T - hogg.Tᵖ
    
    max(jump.i₀ + jump.i₁ / (1 + jump.e₁ * exp(jump.e₂ * ΔT)), 0)
end
