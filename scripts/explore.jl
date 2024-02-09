using JLD2, DotEnv
using DifferentialEquations
using Interpolations

using Plots

using Model

env = DotEnv.config();

begin
    N = 50
    datapath = get(env, "DATAPATH", "data");
    filename = "N=$(N)_Δλ=0.08.jld2"
    termpath = joinpath(datapath, "terminal", filename)
    simpath = joinpath(datapath, "total", filename)
    calpath = joinpath(datapath, "calibration.jld2")
end;

calibration = load_object(calpath);

function valueload(path)
    file = jldopen(path, "r")
	timesteps = keys(file)
	V = Array{Float64}(undef, 50, 50, 50, length(timesteps))
	policy = Array{Policy}(undef, 50, 50, 50, length(timesteps))

	for t in timesteps
		tdx = tryparse(Int, t) + 1
		V[:, :, :, tdx] = file[t]["V"]
		policy[:, :, :, tdx] = file[t]["policy"]
	end
	
	close(file)

    return V, policy
end

model = load(termpath, "model");
Vₘ, policyₘ = valueload(simpath);

# Construct interpolation
timeknots = 0:(size(Vₘ, 4) - 1)
spaceknots = ntuple(i -> range(model.grid.domains[i]...; length = N), 3);

knots = (spaceknots..., timeknots);
V = linear_interpolation(knots, Vₘ);
χ = linear_interpolation(knots, first.(policyₘ), extrapolation_bc = Line());
α = linear_interpolation(knots, last.(policyₘ), extrapolation_bc = Line());

function F!(du, u, parameters, t)
	model, χ, α = parameters
    consumption = χ(u[1], u[2], u[3], t)
    abatement = α(u[1], u[2], u[3], t)

	du[1] = Model.μ(u[1], u[2], model.hogg, model.albedo) / model.hogg.ϵ
	du[2] = Model.γ(t, model.economy, model.calibration) - abatement
    du[3] = Model.b(t, u, Policy(consumption, abatement), model)
end
function G!(du, u, parameters, t)
	model = first(parameters)

	du[1] = model.hogg.σₜ
	du[2] = 0.
    du[3] = model.economy.σₖ
end

fn = SDEFunction(F!, G!);
X₀ = [model.hogg.T₀, log(model.hogg.M₀), log(model.economy.Y₀)];
parameters = (model, χ, α);
prob = SDEProblem(fn, G!, X₀, (0., 1.), parameters);

simlength = 100;
time = Vector{Union{Missing, Float64}}(undef, simlength);
X = Array{Union{Missing, Float64}}(undef, simlength, 3);
p = Array{Union{Missing, Float64}}(undef, simlength, 2);

integrator = init(prob);
for i in 1:100
	step!(integrator)
	Xₜ = integrator.u
	X[i, :] = Xₜ
	time[i] = integrator.t
	consumption = χ(Xₜ[1], Xₜ[2], Xₜ[3], t)
    abatement = α(Xₜ[1], Xₜ[2], Xₜ[3], t)

	p[i, 1] = consumption
	p[i, 2] = abatement
end


