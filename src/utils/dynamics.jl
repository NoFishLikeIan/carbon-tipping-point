mass_matrix(baseline) = diagm([baseline.ϵ / secondtoyears, 1., 1.])

"""
Drift dynamics of (x, m̂, mₛ) given an abatement function α(x, m̂) and a business as usual growth rate g(t).
"""
function F!(du, u, parameters, t)
	# Parameters
	climate, g, α = parameters
	
	x, m̂, mₛ = @view u[1:3]

	du[1] = μ(x, m̂, climate)
	du[2] = g(t) - α(x, m̂)
	du[3] = δₘ(mₛ, first(climate)) * exp(m̂)
end

function G!(du, u, parameters, t)
	baseline = parameters[1][1]	

	du[1] = baseline.σ²ₓ 
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
