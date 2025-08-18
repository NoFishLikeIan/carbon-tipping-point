function printjacobiterminal(ε, iter, magnitude; addition = 2)
    fmt = Printf.Format("Converged in %i iterations, ε = %.$(magnitude + addition)f\n")
    out = Printf.format(fmt, iter, ε)

    print(out); flush(stdout)
end

function printjacobi(ε, iter, maxiter, magnitude; addition = 2)
    fmt = Printf.Format("Iteration %i / %i, ε = %.$(magnitude + addition)f...\r")
    out = Printf.format(fmt, iter, maxiter, ε)

    print(out); flush(stdout)
end

function printbackward!(elapsed, inittime, passcounter, cluster, clustertime)
    nextelapsed = time() - inittime
    Δelapsed = nextelapsed - elapsed
    minutes, seconds = divrem(nextelapsed, 60.)
    n = length(cluster)

    @printf "%2.0fm:%2.0fs (Δ %.2fs): pass %i, cluster size %i, time = %.4f\n" minutes seconds Δelapsed passcounter n clustertime
    flush(stdout)

    elapsed = nextelapsed
end