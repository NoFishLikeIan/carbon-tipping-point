using Revise
using UnPack
using JLD2, DotEnv, CSV
using DataFrames
using DataStructures

using FiniteDiff
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis

using KernelDensity
using Interpolations
using Plots, Printf, PGFPlotsX, Colors
using Statistics

using Model, Grid

include("utils/plotting.jl")
include("utils/saving.jl")

begin # Global variables
    env = DotEnv.config()

    PALETTE = color.(["#003366", "#E31B23", "#005CAB", "#DCEEF3", "#FFC325", "#E6F1EE"])
    SEQPALETTECODE = :YlOrRd
    generateseqpalette(n) = palette(SEQPALETTECODE, n + 2)[3:end]

    LINESTYLE = ["solid", "dashed", "dotted"]
    
    BASELINE_YEAR = parse(Int64, get(env, "BASELINE_YEAR", "2020"))
    DATAPATH = get(env, "DATAPATH", "data")
    PLOTPATH = get(env, "PLOTPATH", "plots")
    PRESENTATIONPATH = joinpath(PLOTPATH, "presentation")
    jumpcolor = RGB(0, 77 / 255, 64 / 255)

    SAVEFIG = false 
    kelvintocelsius = 273.15
end;

begin # Import
    ΔΛ = [0.06, 0.08];
    Ω = [0.002]
	N = 23;
	domains = [
		Hogg().T₀ .+ (0., 9.),
		log.(Hogg().M₀ .* (1., 2.)),
		log.(Economy().Y₀ .* (0.5, 2.))
	]

	preferences = EpsteinZin()
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
    damages = GrowthDamages(ξ = 0.000266, υ = 3.25)

	models = ModelInstance[]
    jumpmodels = ModelBenchmark[]
	

	for Δλ ∈ ΔΛ, ωᵣ ∈ Ω
		economy = Economy(ωᵣ = ωᵣ)
	    albedo = Albedo(λ₂ = 0.31 - Δλ)
		hogg = calibrateHogg(albedo)

	    model = ModelInstance(preferences, economy, damages, hogg, albedo, calibration)
        jumpmodel = ModelBenchmark(preferences, economy, damages, Hogg(), Jump(), calibration)

		push!(models, model)
        push!(jumpmodels, jumpmodel)
	end
end;

G = RegularGrid(domains, N);

function stringtempdev(x::Real; digits = 2)
    fsign = x > 0 ? "+" : ""
    fmt = Printf.Format("$fsign%0.$(digits)f")
    return Printf.format(fmt, x)
end

function makedevxlabels(from, to, model::ModelInstance; step = 0.5, withcurrent = false, digits = 2)

    preindustrialx = range(from, to; step = step)
    xticks = model.hogg.Tᵖ .+ preindustrialx

    xlabels = [stringtempdev(x, digits = digits) for x in preindustrialx]

    if !withcurrent
        return (xticks, xlabels)
    end

    xlabels = [xlabels..., "\$x_0\$"]
    xticks = [xticks..., first(climate).x₀]
    idxs = sortperm(xticks)
    
    return (xticks[idxs], xlabels[idxs])
end

begin # labels and axis
    TEMPLABEL = raw"Temperature deviations $T - T^{\mathrm{p}}$"
    Tspacedev = range(0., 10.; length = 51)
    Tspace = Tspacedev .+ Hogg().Tᵖ
    yearlytime = collect(0:Economy().t₁) 
    ΔTᵤ = last(Tspace) - first(Tspace)
    temperatureticks = makedevxlabels(0., ΔTᵤ, first(models); step = 1, digits = 0)
    presentationtemperatureticks = makedevxlabels(0., ΔTᵤ, first(models); step = 2, digits = 0)
end

begin # Load IPCC data
    IPCCDATAPATH = joinpath(DATAPATH, "climate-data", "proj-median.csv")
    ipccproj = CSV.read(IPCCDATAPATH, DataFrame)

    getscenario(s::Int64) = filter(:Scenario => isequal("SSP$(s) - Baseline"), ipccproj)

    bauscenario = getscenario(5)
    seqpaletteΔλ = Dict(ΔΛ .=> generateseqpalette(length(ΔΛ)))
end

# --- Optimal emissions 
begin # Load simulation
    results = loadtotal(models, G; datapath = DATAPATH)
	jumpresults = loadtotal(jumpmodels, G; datapath = DATAPATH)
end;

begin # Construct interpolations
	ΔT, Δm, Δy = G.domains
	spacenodes = ntuple(i -> range(G.domains[i]...; length = N), 3)

	resultsmap = OrderedDict()
	jumpresultsmap = OrderedDict()

	for ωᵣ ∈ Ω
		j = findfirst(m -> ωᵣ ≈ m.economy.ωᵣ, jumpmodels)

		res = jumpresults[j]	
		ts, V, policy = res
			
		nodes = (spacenodes..., ts)
		χitp = linear_interpolation(nodes, first.(policy); extrapolation_bc = Flat())
		αitp = linear_interpolation(nodes, last.(policy); extrapolation_bc = Flat())
		Vitp = linear_interpolation(nodes, V; extrapolation_bc = Flat())

		
		jumpresultsmap[ωᵣ] = (χitp, αitp, Vitp, jumpmodels[j])
		
		for Δλ ∈ ΔΛ
			k = findfirst(m -> ωᵣ ≈ m.economy.ωᵣ && Δλ ≈ m.albedo.λ₁ - m.albedo.λ₂, models)

			res = results[k]
			model = models[k]
					
			ts, V, policy = res
				
			nodes = (spacenodes..., ts)
			χitp = linear_interpolation(nodes, first.(policy); extrapolation_bc = Flat())
			αitp = linear_interpolation(nodes, last.(policy); extrapolation_bc = Flat())
			Vitp = linear_interpolation(nodes, V; extrapolation_bc = Flat())

			
			resultsmap[(ωᵣ, Δλ)] = (χitp, αitp, Vitp, model)
		end
	end

    Eᵇ = linear_interpolation(calibration.years .- 2020, calibration.emissions, extrapolation_bc = Line());
end;

function F!(dx, x, p::Tuple{ModelInstance, Any, Any}, t)	
	model, χitp, αitp = p
	
	T, m, y = x
	
	χ = χitp(T, m, y, t)
	α = αitp(T, m, y, t)
	
	dx[1] = μ(T, m, model.hogg, model.albedo) / model.hogg.ϵ
	dx[2] = γ(t, model.economy, model.calibration) - α
	dx[3] = b(t, Point(T, m, y), Policy(χ, α), model)

	return
end;
function F!(dx, x, p::Tuple{ModelBenchmark, Any, Any}, t)	
	model, χitp, αitp = p
	
	T, m, y = x
	
	χ = χitp(T, m, y, t)
	α = αitp(T, m, y, t)
	
	dx[1] = μ(T, m, model.hogg) / model.hogg.ϵ
	dx[2] = γ(t, model.economy, model.calibration) - α
	dx[3] = b(t, Point(T, m, y), Policy(χ, α), model)

	return
end;

function G!(dx, x, p, t)
	model = first(p)
	
	dx[1] = model.hogg.σₜ / model.hogg.ϵ
	dx[2] = 0.
	dx[3] = model.economy.σₖ
	
	return
end;

begin # Solver
    ω = first(keys(resultsmap))[1]
	tspan = (0., Economy().t₁)

	problems = OrderedDict{Float64, SDEProblem}()
	fn = SDEFunction(F!, G!)

	for Δλ ∈ ΔΛ
		χitp, αitp, _, model = resultsmap[(ω, Δλ)]
		parameters = (model, χitp, αitp)
		
		problems[Δλ] = SDEProblem(fn, X₀, tspan, parameters)
	end	

    solutions = OrderedDict(key => solve(EnsembleProblem(prob), EnsembleDistributed(); trajectories = 30) for (key, prob) ∈ problems);
end;

begin # Solver for benchmark
	χitp, αitp, _, model = jumpresultsmap[ω]
	parameters = (model, χitp, αitp)
			
	jumpproblem = SDEProblem(fn, X₀, tspan, parameters)
	jumpsolution = solve(EnsembleProblem(jumpproblem), EnsembleDistributed(), trajectories = 30)
end;

begin # Data extraction
    timespan = range(0, 80; step = 0.5)
    function emissionpath(solution, model, αitp)
        E = Matrix{Float64}(undef, length(timespan), length(solution))
    
        for (j, sim) ∈ enumerate(solution), (i, tᵢ) ∈ enumerate(timespan)
            T, m, y = sim(tᵢ)
            αₜ = αitp(T, m, y, tᵢ)
            
            M = exp(m)
            Eₜ = (M / Model.Gtonoverppm) * (γ(tᵢ, model.economy, model.calibration) - αₜ)
    
            
            E[i, j] = Eₜ
        end
    
        return E
    end;
    function consumptionpath(solution, model, χitp)
        C = Matrix{Float64}(undef, length(timespan), length(solution))
    
        for (j, sim) ∈ enumerate(solution), (i, tᵢ) ∈ enumerate(timespan)
            T, m, y = sim(tᵢ)
            χᵢ = χitp(T, m, y, tᵢ)
            
            C[i, j] = exp(y) * χᵢ
        end
    
        return C
    end;
    function variablepath(solution, model)
        X = Matrix{Point}(undef, length(timespan), length(solution))
    
        for (j, sim) ∈ enumerate(solution), (i, tᵢ) ∈ enumerate(timespan)
            T, m, y = sim(tᵢ)
            X[i, j] = Point(T, m, y)
        end
    
        return X
    end;
end

begin # Emission comparison figure
    decadeticks = 0:10:80

    emissionfig = @pgf Axis(
        {
            width = raw"\linewidth",
            height = raw"0.7\linewidth",
            grid = "both",
            xlabel = raw"Year",
            ylabel = raw"Net Emissions",
            xmin = minimum(timespan), xmax = maximum(timespan),
            xtick = decadeticks, xticklabels = decadeticks .+ 2020,
            ymin = -1., ymax = 25.,
        }
    )
    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-optemissions.tikz"), emissionfig; include_preamble = true) 
    end

        
    # Albedo
    for Δλ ∈ [0.06, 0.08]
        _, αitp, _, model = resultsmap[(ω, Δλ)];
        solution = solutions[Δλ];
        emissions = emissionpath(solution, model, αitp);

        label = @sprintf("\$\\Delta\\lambda = %.0f \\%%\$", 100 * Δλ)

        for E ∈ eachcol(emissions)
            @pgf push!(
                emissionfig, 
                Plot({forget_plot, color = seqpaletteΔλ[Δλ], opacity = 0.2}, Coordinates(zip(timespan, E))
                )
            )
        end

        @pgf push!(
            emissionfig, 
            Plot({ color = seqpaletteΔλ[Δλ], line_width = "3pt" },
                Coordinates(zip(timespan, emissions))
            ), LegendEntry(label)
        )
    end

    # Jump
    jumpcolor = RGB(0, 77 / 255, 64 / 255)
	_, αitp, _, jumpmodel = jumpresultsmap[ω]
	emissions = emissionpath(jumpsolution, jumpmodel, αitp)

    for E ∈ eachcol(emissions)
        @pgf push!(
            emissionfig, 
            Plot({forget_plot, color = jumpcolor, opacity = 0.2}, Coordinates(zip(timespan, E))
            )
        )
    end

    @pgf push!(
        emissionfig, 
        Plot({ color = jumpcolor, line_width = "3pt" },
            Coordinates(zip(timespan, median(emissions, dims = 2)))
        ), LegendEntry("Stochastic")
    )

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "optemissions.tikz"), emissionfig; include_preamble = true) 
    end


    emissionfig
end

begin # Consumption
    consumptionfig = @pgf Axis(
        {
            width = raw"\linewidth",
            height = raw"0.7\linewidth",
            grid = "both",
            xlabel = raw"Year",
            ylabel = raw"GDP / Consumption [Trillions US \$]",
            no_markers,
            ultra_thick,
            xmin = minimum(timespan), xmax = maximum(timespan),
            xtick = decadeticks, xticklabels = decadeticks .+ 2020,
            ymin = 0., ymax = 200.,
        }
    )

    for Δλ ∈ [0.06, 0.08]
        χitp, _, _, model = resultsmap[(ω, Δλ)];
        solution = solutions[Δλ];

        X = variablepath(solution, model)
        Y = [exp(x.y) for x ∈ X]

        label = @sprintf("\$\\Delta\\lambda = %.0f \\%%\$", 100 * Δλ)

        @pgf push!(
            consumptionfig, 
            Plot({ color = seqpaletteΔλ[Δλ], line_width = "3pt" },
                Coordinates(zip(timespan, median(Y, dims = 2)))
            ), LegendEntry(label)
        )
    end
    
    # Jump
	jumpχitp, _, _, jumpmodel = jumpresultsmap[ω]
	X = variablepath(jumpsolution, jumpmodel)
	Y = [exp(x.y) for x ∈ X]
    

    @pgf push!(
        consumptionfig, 
        Plot({ color = jumpcolor, line_width = "3pt" },
            Coordinates(zip(timespan, median(Y, dims = 2)))
        ), LegendEntry("Stochastic")
    )

    @pgf consumptionfig["legend style"] = raw"at = {(0.9, 0.3)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "optgdp.tikz"), consumptionfig; include_preamble = true) 
    end

    consumptionfig
end

begin # Temperature
    ytick, yticklabels = makedevxlabels(1, 2, first(models); step = 0.25, digits = 2)

    tempfig = @pgf Axis(
        {
            width = raw"\linewidth",
            height = raw"0.7\linewidth",
            grid = "both",
            xlabel = raw"Year",
            ylabel = TEMPLABEL,
            xmin = minimum(timespan), xmax = maximum(timespan),
            xtick = decadeticks, xticklabels = decadeticks .+ 2020, 
            ytick = ytick, yticklabels = yticklabels
        }
    )

    for Δλ ∈ [0.06, 0.08]
        χitp, _, _, model = resultsmap[(ω, Δλ)];
        solution = solutions[Δλ];

        X = variablepath(solution, model)
        temps = [x.T for x ∈ X]

        label = @sprintf("\$\\Delta\\lambda = %.0f \\%%\$", 100 * Δλ)

        @pgf push!(
            tempfig, 
            Plot({ color = seqpaletteΔλ[Δλ], line_width = "3pt" },
                Coordinates(zip(timespan, median(temps, dims = 2)))
            ), LegendEntry(label)
        )

    end
    
    # Jump
	jumpχitp, _, _, jumpmodel = jumpresultsmap[ω]
	X = variablepath(jumpsolution, jumpmodel)
	temps = [x.T for x ∈ X]

    @pgf push!(
        tempfig, 
        Plot({ color = jumpcolor, line_width = "3pt" },
            Coordinates(zip(timespan, median(temps, dims = 2)))
        ), LegendEntry("Stochastic")
    )

    @pgf tempfig["legend style"] = raw"at = {(0.9, 0.3)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "opttemp.tikz"), tempfig; include_preamble = true) 
    end

    tempfig
end
