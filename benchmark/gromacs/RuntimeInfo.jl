using SHA, TOML

function benchmark_runtime_info()
    manifest = joinpath(dirname(Base.active_project()), "Manifest.toml")
    packages = Dict{String,String}()
    for (id, mod) in Base.loaded_modules
        if id.name in ("CUDA", "CUDACore", "GPUCompiler", "GPUArrays", "ParticleDynamics")
            version = pkgversion(mod)
            version === nothing || (packages[id.name] = string(version))
        end
    end
    return Dict("julia" => string(VERSION), "project" => Base.active_project(),
                "depot" => copy(DEPOT_PATH), "packages" => packages,
                "manifest_sha256" => isfile(manifest) ? bytes2hex(sha256(read(manifest))) : "absent")
end

function print_benchmark_runtime()
    println("Loaded runtime provenance:")
    TOML.print(stdout, benchmark_runtime_info(); sorted=true)
    println()
end
