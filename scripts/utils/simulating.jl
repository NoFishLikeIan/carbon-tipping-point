using Model

function Fbau!(du, u, model::ModelInstance, t)
	du[1] = μ(u[1], u[2], model.hogg, model.albedo) / model.hogg.ϵ
	du[2] = γ(t, model.economy, model.calibration)
end
function Fbau!(du, u, model::ModelBenchmark, t)
    du[1] = μ(u[1], u[2], model.hogg) / model.hogg.ϵ
	du[2] = γ(t, model.economy, model.calibration)
end
function Gbau!(du, u, model, t)    
	du[1] = model.hogg.σₜ / model.hogg.ϵ
	du[2] = model.hogg.σₘ
end

function rate(u, model::ModelBenchmark, t)
    intensity(u[1], model.hogg, model.jump)
end
function affect!(integrator)
    model = integrator.p
    q = increase(integrator.u[1], model.hogg, model.jump)
    integrator.u[1] += q
end