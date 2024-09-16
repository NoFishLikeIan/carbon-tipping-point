using Revise
using Model

using Plots

β₈₀ = 2.7 / 100
ω₀ = 22 / 100
ωᵣ = log(ω₀ / β₈₀) / 80.

economy = Economy(ω₀ = ω₀, ωᵣ = ωᵣ)

timespan = range(0, 80.; length = 101)
begin
    βfig = plot(; xlabel = "\$t\$", ylabel = "perc. of GDP")

    for eps in [.17, .27, .46, 1.]
        βcal = t -> Model.β(t, eps, economy)

        plot!(βfig, timespan, βcal; label = "ε = $eps")
    end

    βfig
end