mass_matrix(baseline) = diagm([baseline.ϵ / secondtoyears, 1., 1.])

"""
Drift dynamics of (T, m, N) given an abatement function γᵇ(T, m) and a business as usual growth rate g(t).
"""
function F!(du, u, parameters, t)
	# Parameters
	climate, γᵇ, α = parameters
	
	T, m, N = @view u[1:3]
	M = exp(m)

	du[1] = μ(T, m, climate)
	du[2] = γᵇ(t) - α(T, m)
	du[3] = δₘ(N, first(climate)) * M
end

function G!(du, u, parameters, t)
	baseline = parameters[1][1]	

	du[1] = baseline.σ²ₜ 
	du[2] = baseline.σ²ₘ
	du[3] = 0.
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
