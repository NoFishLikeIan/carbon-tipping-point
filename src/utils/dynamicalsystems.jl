function simulateclimatepath(
	σₓ::Real, γ::Real, m::MendezFarazmand, e::Function; 
	T = 80, ntraj = 1000
)
	function F!(du, u, p, t)
		γ, σₓ = p
		x, c = u
	
		du[1] = μ(x, max(c, m.cₚ), m)
		du[2] = e(x, c, σₓ, γ) - m.δ * c
	end

	function G!(du, u, p, t)
		σₓ = p[2]
		du[1] = σₓ
		du[2] = 0.
	end

	prob = SDEProblem(F!, G!, [m.x₀, m.c₀], (0, T), [γ, σₓ])
	ensprob = EnsembleProblem(prob)
	sim = solve(ensprob, SRIW1(), trajectories = ntraj)
	
	return sim
end

function computeoptimalemissions(σₓ, γ, sim, e::Function; Tsim = 1001)
	T = first(sim).t |> last
	timespan = range(0, T; length = Tsim)
	nsim = size(sim, 3)

	optemissions = Matrix{Float64}(undef, Tsim, nsim)

	for idxsim in 1:nsim
		optemissions[:, idxsim] .= [e(x, c, σₓ, γ) for (x, c) in sim[idxsim](timespan).u]
	end

	return optemissions
end

function extractquartiles(ensamblesim, quartile; Tsim = 2001)
	T = first(ensamblesim).t |> last
	timespan = range(0, T; length = Tsim)
	upperq(t) = EnsembleAnalysis.timepoint_meanvar(ensamblesim, t)[2][1]
	lowerq(t) = EnsembleAnalysis.timepoint_quantile(ensamblesim, 1 - quartile, t)[1]
	upperq(t) = EnsembleAnalysis.timepoint_quantile(ensamblesim, quartile, t)[1]
		
	mediansim = hcat([EnsembleAnalysis.timepoint_median(ensamblesim, t) for t in timespan]...)'
	lowerqsim = lowerq.(timespan)
	upperqsim = upperq.(timespan)

	return lowerqsim, mediansim, upperqsim	
end