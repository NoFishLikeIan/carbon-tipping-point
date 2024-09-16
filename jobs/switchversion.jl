using Pkg

const relativedirs = ["src/Grid", "src/Model", ""]; # Order is important
const basepath = dirname(Base.active_project())

modulesdir = joinpath.(basepath, relativedirs)

@assert all(isdir.(modulesdir))

for dir in modulesdir
    manifestfile = joinpath(dir, "Manifest.toml")

    if isfile(manifestfile)
        rm(manifestfile)
    end
end

for (k, dir) in enumerate(modulesdir)
    println("Working on $dir")
    Pkg.activate(dir)

    localdeps = modulesdir[1:(k - 1)]

    pkgdeps = @. last(splitdir(localdeps))

    for pkg in pkgdeps
        Pkg.rm(pkg)
    end

    for deps in localdeps
        Pkg.develop(path = deps)
    end

    Pkg.instantiate()
    Pkg.precompile()
end