using PGFPlotsX
using Plots
using Roots
using LaTeXStrings
using DifferentialEquations

using Colors, ColorSchemes

PALETTE = colorschemes[:grays];
plotpath = "plots/toy-model"

colors = get(PALETTE, [0., 0.7]);
LINE_WIDTH = 2.5

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usetikzlibrary{arrows.meta}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{siunitx}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\ppm}{p.p.m.}")

function λ(T; a = 0.08, c = 1 / 2)
    a * T^2  - 3 * √(a * c) * T + 2c
end

function λ′(T; a = 0.08, c = 1 / 2)
    2a * T - 3 * √(c * a)
end

function f(T, m; params...)
    m - λ(T; params...) * T
end

function flinear(T, m)
    m - λ(0) * T
end

function f′(T, m; params...)
    -λ′(T; params...) * T - λ(T; params...)
end

function F!(du, u, p, t)
    a, E = p

    du[1] = f(u[1], log(u[2]); a = a)
    du[2] = E
end

# Plot 
begin
    a = 0.08
    label = a > 0. ? "feedback" : "linear"
    u₀ = [0., 1.];
    E = 1.

    timespan = (0., 10.)

    prob = ODEProblem{true}(F!, u₀, timespan, (a, E));
    sol = solve(prob)

    Tspace = range(0, 7.; length = 51)
    Tticks = range(extrema(Tspace)...; step = 1.)
    Ttickslabel = [L"$%$(Int(T))^\circ$" for T in Tticks]

    timesteps = range(timespan...; length = 51)
    Ttraj = first.(sol(timesteps).u)
    
    trajaxis = @pgf Axis({
        xlabel = L"Time $t$", xmin = minimum(timesteps), xmax = maximum(timesteps),
        ylabel = L"Temperature $T_t \; [\si{\degree}]$", ymin = minimum(Tspace), ymax = maximum(Tspace),
        grid = "both",
        ytick = Tticks, yticklabels = Ttickslabel,
        width = raw"0.595\linewidth", height = raw"0.425\linewidth",
        xtick = range(extrema(timesteps)...; step = 1.)
    })

    if a > 0.
        prob = ODEProblem{true}(F!, u₀, timespan, (0., E));
        sol = solve(prob)

        trajplot = @pgf Plot({
            line_width = LINE_WIDTH, color = colors[1], dotted
        }, Coordinates(timesteps, first.(sol(timesteps).u)))

        push!(trajaxis, trajplot, LegendEntry(raw"\footnotesize Linear"))
    end

    trajplot = @pgf Plot({
        line_width = LINE_WIDTH, color = colors[1]
    }, Coordinates(timesteps, Ttraj))

    push!(trajaxis, trajplot, LegendEntry(raw"\footnotesize Feedback"))

    @pgf trajaxis["legend style"] = raw"at = {(0.4, 0.98)}"

    PGFPlotsX.save(joinpath(plotpath, "trajfig_feedback.tikz"), trajaxis; include_preamble = true)

    trajaxis
end

begin # Phase diagram
    mratios = [1., 1.5, 2.]

    Textrema = a > 0 ? (-1., 6.) : (-1., 2.)
    Tspace = range(Textrema...; length = 51)
    Tticks = range(Textrema...; step = 1)

    labelstep = a > 0. ? 2 : 1
    Ttickslabel = [
        !isodd(i) ? L"$%$(Int(T))^\circ$" : "" 
        for (i, T) in enumerate(Tticks)
    ]

    defaxisopts = @pgf {
        xmin = Textrema[1], xmax = Textrema[2],
        xtick = Tticks, xticklabels = Ttickslabel, 
        ymin = -1, ymax = 1.5,
        grid = "both",
        width = raw"0.34\linewidth", 
        height = raw"0.34\linewidth"
    }

    gp = @pgf GroupPlot({
        group_style = {group_size = "$(length(mratios)) by 1", horizontal_sep = "0.5em"}
    })

    for (i, mratio) in enumerate(mratios)

        title = if mratio ≈ 1
            L"$M_t = M^p$"
        elseif isinteger(mratio)
            L"$M_t = %$(Int(mratio))M^p$"
        else
            r = Rational(mratio)
            L"$M_t = \frac{%$(r.num)}{%$(r.den)}M^p$"
        end
        
        axis = @pgf Axis({
            defaxisopts...,
            ylabel = i > 1 ? "" : L"\small Change in temperature $\mathrm{d}T_t \big/ \mathrm{d} t$", 
            ytick = -1:0.5:2,
            xlabel = i == 2 ? L"\small Temperature $T_t \; [\si{\degree}]$" : "",
            yticklabels = raw"\empty",
            title = title
        })

        @pgf push!(axis, HLine({ line_width = 1.5 }, 0.))

        m = log(mratio)

        fₐ = T -> f(T, m; a)
        f′ₐ = T -> f′(T, m; a)

        curve = @pgf Plot({
            line_width = 2.,
        }, Coordinates(Tspace, fₐ.(Tspace)))        
        @pgf push!(axis, curve)

        (i == 1) && @pgf push!(axis, LegendEntry(raw"\footnotesize Feedback"))

        line = @pgf Plot({
            line_width = 2., dotted
        }, Coordinates(Tspace, flinear.(Tspace, m)))
        @pgf push!(axis, line)

        (i == 1) && @pgf push!(axis, LegendEntry(raw"\footnotesize Linear"))
        

        steadystates = find_zeros(fₐ, (-1., 8.))

        for T̄ in steadystates
            markercolor = f′ₐ(T̄) < 0. ? 
                (T̄ < 3 ? colors[2] : colors[1]) : 
                "white"

            strokecolor = f′ₐ(T̄) < 0. ? 
                (T̄ < 3 ? colors[2] : colors[1]) : 
                "black"

            options = @pgf {
                mark_options = {fill = markercolor, stroke = strokecolor, scale = 1.5},
                forget_plot,
                only_marks
            }

            point = Coordinates([(T̄, 0.)])

            push!(axis, Plot(options, point))
        end

        for T in range(extrema(Tspace)...; step = 1.)
            if abs(fₐ(T)) < 1e-3
                continue
            end

            Tend = T + sign(fₐ(T)) * 1 / 4

            points = Coordinates([(T, 0.), (Tend, 0.)])

            @pgf push!(axis, Plot({"-{Latex[length=3mm]}", forget_plot}, points))
        end

        @pgf push!(gp, axis)
    end

    PGFPlotsX.save(joinpath(plotpath, "tipping_illustration_$label.tikz"), gp; include_preamble = true)

    gp
end

# Optimisation of τ
d(T) = exp(T)
T̄linear(τ) = log(1 + E * τ) / λ(0)
function Jlinear(τ; ρ = 0.1, β = 2., c = 1., params...)
    T̄ = T̄linear(τ)
    w = exp(-ρ * τ)

    β * E * (1 - w) - c * d(T̄) / ρ
end
begin # Plot marginal benefit and marginal cost
    a = 0.02
    timesteps = range(0., 3., length = 101)

    objvalues = Jlinear.(timesteps; β = 4.7, c = 2.9, ρ = 1.)

    Jfig = @pgf Axis({
        xmin = 0., xmax = maximum(timesteps),
        xlabel = L"Emissions stopping time $\tau$", grid = "both",
        yticklabels = raw"\empty", ylabel = L"Social objective $J(\tau)$",
        scaled_y_ticks = false,
    })


    coords = Coordinates(timesteps, objvalues)

    fig = @pgf Plot({ line_width = LINE_WIDTH, color = colors[2] }, coords)

    push!(Jfig, fig)

    PGFPlotsX.save(joinpath(plotpath, "maximisation_feedback_linear.tikz"), Jfig; include_preamble = true)

    Jfig
end

Tsteadystate(τ; params...) = find_zeros(
    T -> T * λ(T; params...) - log(1 + E * τ),
    (0., 100.)
)

function J(τ; ρ = 0.1, β = 2., c = 1., params...)
    steadystate = Tsteadystate(τ; params...)
    w = exp(-ρ * τ)

    [
        β * E * (1 - w) - w * c * d(T̄) for T̄ in steadystate
    ]
end

begin # Plot marginal benefit and marginal cost
    a = 0.02
    timesteps = range(0., 3., length = 101)

    objvalues = J.(timesteps; a, ρ = 0.1, β = 30_000., c = 1.)

    bifurcation = findlast(v -> length(v) > 1, objvalues)

    triple = objvalues[1:bifurcation]
    single = objvalues[(bifurcation + 1):end]

    Jfig = @pgf Axis({
        xmin = 0., xmax = maximum(timesteps),
        xlabel = L"Emissions stopping time $\tau$", grid = "both",
        yticklabels = raw"\empty", ylabel = L"Social objective $J(\tau)$",
        scaled_y_ticks = false,
    })

    upperpath = getindex.(triple, 1)
    lowerpath = last.(objvalues)

    uppercoords = Coordinates(timesteps[1:bifurcation], upperpath)

    upperplot = @pgf Plot({ line_width = LINE_WIDTH, color = colors[2] }, uppercoords)

    uppermark = @pgf Plot({ only_marks, mark_options = {scale = 1.25}, forget_plot, color = colors[2] }, Coordinates(timesteps[[bifurcation]], upperpath[[bifurcation]]))

    lowerplot = @pgf Plot({line_width = LINE_WIDTH, color = colors[1]}, Coordinates(timesteps, lowerpath))

    push!(Jfig, upperplot, uppermark, LegendEntry(raw"\footnotesize Low temp."), lowerplot, LegendEntry(raw"\footnotesize High temp."))

    PGFPlotsX.save(joinpath(plotpath, "maximisation_feedback.tikz"), Jfig; include_preamble = true)

    Jfig
end
