mass_matrix(baseline) = diagm([baseline.ϵ / secondtoyears, 1.])

"""
Drift dynamics of (T, m, N) given an abatement function γᵇ(T, m) and a business as usual growth rate g(t).
"""
function Fext!(du, u, parameters, t)
	hogg, albedo, γᵇ, α = parameters
	
	T, m, N = @view u[1:3]
	M = exp(m)

	du[1] = μ(T, m, hogg, albedo)
	du[2] = γᵇ(t) - α(T, m)
	du[3] = δₘ(N, hogg) * M
end

function Gext!(du, u, parameters, t)
	hogg = first(parameters)

	du[1] = hogg.σ²ₜ 
	du[2] = 0.
	du[3] = 0.
end

function F!(du, u, parameters, t)
	hogg, albedo, γᵇ, α = parameters
	du[1] = μ(u[1], u[2], hogg, albedo)
	du[2] = γᵇ(t) - α(u[1], u[2])
end


function G!(du, u, parameters, t)
	hogg = first(parameters)

	du[1] = hogg.σ²ₜ 
	du[2] = 0.
end


function extractoptimalemissions(σₓ, sim, e::Function; Tsim = 1001)
	T = first(sim).t |> last
	timespan = range(0, T; length = Tsim)
	nsim = size(sim, 3)

	optemissions = Matrix{Float64}(undef, Tsim, nsim)

	for idxsim in 1:nsim
		optemissions[:, idxsim] .= [e(x, m, σₓ) for (x, m) in sim[idxsim](timespan).u]
	end

	return optemissions
end
