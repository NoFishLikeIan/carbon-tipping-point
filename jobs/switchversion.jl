using Pkg

const modulesdir = ["src/Grid", "src/Model", "."]; # Order is important

for dir in modulesdir
    manifestfile = joinpath(dir, "Manifest.toml")

    if isfile(manifestfile)
        rm(manifestfile)
    end
end

for (k, dir) in enumerate(modulesdir)
    Pkg.activate(dir)

    localdeps = modulesdir[1:(k - 1)]
    for deps in localdeps
        Pkg.develop(path = deps)
    end

    Pkg.instantiate()
    Pkg.precompile()
end