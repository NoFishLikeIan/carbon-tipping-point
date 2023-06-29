function simulateclimatepath(
	σₓ::Real, climate::MendezFarazmand, e::Function; 
	T = 80, ntraj = 1000
)
	function F!(du, u, σₓ, t)
		x, m = u
	
		du[1] = μ(x, max(m, climate.mₚ), climate)
		du[2] = e(x, m, σₓ) - climate.δ * m
	end

	function G!(du, u, σₓ, t)
		du[1] = σₓ
		du[2] = 0.
	end

	prob = SDEProblem(F!, G!, [climate.x₀, climate.mₚ], (0, T), σₓ)
	ensprob = EnsembleProblem(prob)
	sim = solve(ensprob, SRIW1(), trajectories = ntraj)
	
	return sim
end

function computeoptimalemissions(σₓ, sim, e::Function; Tsim = 1001)
	T = first(sim).t |> last
	timespan = range(0, T; length = Tsim)
	nsim = size(sim, 3)

	optemissions = Matrix{Float64}(undef, Tsim, nsim)

	for idxsim in 1:nsim
		optemissions[:, idxsim] .= [e(x, m, σₓ) for (x, m) in sim[idxsim](timespan).u]
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