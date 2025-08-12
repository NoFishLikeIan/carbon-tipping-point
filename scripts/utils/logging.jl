function printjacobiterminal(ε, iter, magnitude; addition = 2)
    fmt = Printf.Format("Converged in %i iterations, ε = %.$(magnitude + addition)f\n")
    out = Printf.format(fmt, iter, ε)

    print(out)
end

function printjacobi(ε, iter, maxiter, magnitude; addition = 2)
    fmt = Printf.Format("Iteration %i / %i, ε = %.$(magnitude + addition)f...\r")
    out = Printf.format(fmt, iter, maxiter, ε)

    print(out)
end