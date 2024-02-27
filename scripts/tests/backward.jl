using Revise
using Test: @test
using BenchmarkTools: @btime
using JLD2
using Plots

includet("../backward.jl")
includet("../utils/plotting.jl")
includet("../utils/saving.jl")


begin
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	preferences = EpsteinZin()
end

begin
	N = 21

	Tdomain = hogg.T₀ .+ (0., 7.)
	mdomain = mstable.(Tdomain, Ref(hogg), Ref(Albedo()))
	ydomain = log.(economy.Y₀ .* (0.5, 2.))
	
	domains = [Tdomain, mdomain, ydomain]

	G = RegularGrid(domains, N);
end

# --- Albedo
model = ModelInstance(preferences, economy, hogg, Albedo(), calibration);

V̄, terminalpolicy = loadterminal(model, G);


# --- Jump
model = ModelBenchmark(preferences, economy, hogg, Jump(), calibration);
V̄, terminalpolicy = loadterminal(model, G);

policy = [Policy(χ, 0.) for χ ∈ terminalpolicy[:, :, :, 1]];
V = deepcopy(V̄[:, :, :, 1]);

computevalue(model, G; cache = false, verbose = true)