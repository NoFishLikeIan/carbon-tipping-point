using Revise
using JLD2, DotEnv, CSV, UnPack
using FastClosures

using Plots
using PGFPlotsX
using LaTeXStrings, Printf, Colors, ColorSchemes
using Contour

using Interpolations: Extrapolation

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")
default(fontfamily = "Computer Modern",
    linewidth = 2, framestyle = :box, 
    label = nothing, grid = false)    
scalefontsizes(1.3)

using Statistics
using Model, Grid

include("utils.jl")
include("../utils/saving.jl")
include("../utils/simulating.jl")

begin # Environment variables
    env = DotEnv.config()
    datapath = get(env, "DATAPATH", "data")
    simulationpath = get(env, "SIMULATIONPATH", "simulations")

    BASELINE_YEAR = 2020
    SAVEFIG = false

    calibration = load_object(joinpath(datapath, "calibration.jld2"))

    experimentpath = joinpath(datapath, "experiments", "negative.jld2")
    @assert ispath(experimentpath)

    regretpath = joinpath(datapath, "experiments", "regret-negative.jld2")
    @assert ispath(regretpath)
end;

begin # Load simulations and interpolations 
    @load experimentpath solutions;
    allmodels = collect(keys(solutions));
    tippingmodels = filter(model -> model isa TippingModel, allmodels);
    sort!(tippingmodels; by = model -> model.albedo.Tᶜ);

    jumpmodels = filter(model -> model isa JumpModel, allmodels);
    models = AbstractModel[tippingmodels..., jumpmodels...];

    results = loadtotal.(models; datapath = joinpath(datapath, simulationpath))
    
    interpolations = buildinterpolations.(results)
    itpsmap = Dict{AbstractModel, Dict{Symbol, Extrapolation}}(models .=> interpolations)
end;

begin # Labels, colors and axis
    thresholds = sort([model.albedo.Tᶜ for model in tippingmodels])

    PALETTE = colorschemes[:grays]
    graypalette = n -> n > 1 ? get(PALETTE, range(0.1, 0.8; length = n)) : 0.8

    thresholdscolors = Dict(thresholds .=> graypalette(length(thresholds)))

    rawlabels = [ "Imminent", "Remote", "Benchmark"]
    labels = Dict{AbstractModel, String}(models .=> rawlabels)

    TEMPLABEL = L"Temperature deviations $T_t - T^{p}$"
    defopts = @pgf { line_width = 2.5 }

    ΔTmax = 8.
    ΔTspace = range(0., ΔTmax; length = 51)
    Tspace = ΔTspace .+ Hogg().Tᵖ
    Tmin, Tmax = extrema(Tspace)

    mmin, mmax = mstable.(extrema(Tspace), Hogg())
    mspace = range(mmin, mmax; length = 51)

    horizon = round(Int64, last(calibration.tspan))
    yearlytime = range(0., horizon; step = 1 / 3) |> collect

    temperatureticks = makedeviationtickz(0., ΔTmax, first(models); step = 1, digits = 0)
end;

# Contour plot
model = last(tippingmodels)
α = itpsmap[model][:α]
emissivity = @closure (T, m, t) -> ε(t, exp(m), α(T, m, t), model)

contourf(mspace, Tspace, (m, T) -> α(T, m, 80.), linewidth = 0., margins = 5Plots.mm, c = :coolwarm)