const secondstoyears = 60 * 60 * 24 * 365.25
const Gtonoverppm = 1 / 7.821

Base.@kwdef struct Albedo
    Tₐ::Float64 = 3 # Transition rate
	T₁::Float64 = 290.5 # Pre-transition temperature
    T₂::Float64 = 291. # Post-transition temeprature 
     
    λ₁::Float64 = 0.31 # Pre-transition albedo
    λ₂::Float64 = 0.23 # Post-transition albedo
end

Base.@kwdef struct Jump
    j₂::Float64 = -0.0029
    j₁::Float64 = 0.0568
    j₀::Float64 = -0.0577

    i₀::Float64 = -1/4
    i₁::Float64 = 0.95
    
    e₁::Float64 = 2.8
    e₂::Float64 = -0.3325
end

Base.@kwdef struct Hogg
    # Current and pre-industrial data temperature and carbon concentration
    T₀::Float64 = 288.29 # [K]
    Tᵖ::Float64 = 287.15 # [K]
    M₀::Float64 = 410 # [p.p.m.]
    Mᵖ::Float64 = 280 # [p.p.m.]

    N₀::Float64 = 286.65543 # [p.p.m.]
    
    σₜ::Float64 = 1.585 # Volatility of temperature

    # Climate sensitwivity
    S₀::Float64 = 340.5 # [W / m²] Mean solar radiation

    ϵ::Float64 = 5e8 / secondstoyears # years * [J / m² K] / s Heat capacity of the ocean
    η::Float64 = 5.67e-8 # Stefan-Boltzmann constant 
    
    G₁::Float64 = 20.5 # [W / m²] Effect of CO₂ on radiation budget
    G₀::Float64 = 150 # [W / m²] Pre-industrial GHG radiation budget
    
    # Decay rate of carbon concentration
    aδ::Float64 = 0.0176
    bδ::Float64 = -27.36
    cδ::Float64 = 314.8
end

"Obtain an Hogg calibration consistent with the Albedo calibration"
function calibrateHogg(albedo::Albedo; b = (330., 380.))::Hogg
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
"Heaviside function"
H(T, Tₐ) = (1 + tanh(T / Tₐ)) / 2
H(T, albedo::Albedo) = H(T, albedo.Tₐ)
sigmoid(x) = inv(1 + exp(-x))

"Transition function"
function L(T, albedo::Albedo)
    Tᵢ = (albedo.T₁ + albedo.T₂) / 2
    return sigmoid(T - Tᵢ)
end

"Albedo coefficient"
λ(T, albedo::Albedo) = albedo.λ₁ - (albedo.λ₁ - albedo.λ₂) * L(T, albedo)

"Radiation dynamics"
fₜ(T, hogg::Hogg, albedo::Albedo) = hogg.S₀ * (1 - λ(T, albedo)) - hogg.η * T^4
fₜ(T, hogg::Hogg) = hogg.S₀ * 0.69 - hogg.η * T^4

"CO2 forcing given log of CO2 concentration"
fₘ(m, hogg::Hogg) = hogg.G₀ + hogg.G₁ * (m - log(hogg.Mᵖ))
fₘ⁻¹(r, hogg::Hogg) = log(hogg.Mᵖ) + (r - hogg.G₀) / hogg.G₁


"Drift temperature dynamics"
μ(T, m, hogg::Hogg, albedo::Albedo) = fₜ(T, hogg, albedo) + fₘ(m, hogg)
μ(T, m, hogg::Hogg) = fₜ(T, hogg) + fₘ(m, hogg)

"Compute CO₂ concentration consistent with temperature T"
mstable(T, hogg::Hogg, albedo::Albedo) = fₘ⁻¹(-fₜ(T, hogg, albedo), hogg)
mstable(T, hogg::Hogg) = fₘ⁻¹(-fₜ(T, hogg), hogg)
Mstable(T, args...) = exp(mstable(T, args...))


function potential(T, m, hogg::Hogg, albedo::Albedo)
	@unpack λ₁, λ₂ = albedo
    Tᵢ = (albedo.T₁ + albedo.T₂) / 2
	G = Model.fₘ(m, hogg)

	(hogg.η / 5) * T^5 - G * T - (1 - λ₁) * hogg.S₀ * T - hogg.S₀ * (λ₁ - λ₂) * log(1 + exp(T - Tᵢ))
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
    
    jump.i₀ + jump.i₁ / (1 + jump.e₁ * exp(jump.e₂ * ΔT))
end