#!/usr/bin/env julia

using DelimitedFiles
using PyPlot
using Printf
using Statistics
using LaTeXStrings

function parse_tail_stats(path::AbstractString; n::Int=4)
    # Return mean and std of the last n data lines for E_pot and cc
    io = open(path, "r"); lines = readlines(io); close(io)
    if length(lines) < 2
        return nothing
    end
    # Extract data lines (skip header line 1); keep non-empty
    data_lines = [ln for ln in lines[2:end] if !isempty(strip(ln))]
    if isempty(data_lines)
        return nothing
    end
    k = min(n, length(data_lines))
    tail = data_lines[end-k+1:end]
    epots = Float64[]
    ccs = Float64[]
    for line in tail
        parts = split(line, '|')
        if length(parts) < 13; continue; end
        vals = map(s -> try parse(Float64, strip(s)) catch; NaN end, parts)
        push!(epots, vals[3])
        push!(ccs, vals[11])
    end
    if isempty(epots)
        return nothing
    end
    return (E_pot=mean(epots), cc=mean(ccs), E_pot_err=std(epots), cc_err=std(ccs))
end

function parse_dt_eps_from_filename(fname::AbstractString)
    # Expect: traj2d_singleT_VV_homo_soft_phi_eps_dt.log
    m = match(r"soft_([0-9.]+)_([0-9.]+e[+-]?\d+)_([0-9.]+e[+-]?\d+)\.log$", fname)
    m === nothing && return nothing
    ϕ = parse(Float64, m.captures[1])
    ε = parse(Float64, m.captures[2])
    dt = parse(Float64, m.captures[3])
    return (phi=ϕ, eps=ε, dt=dt)
end

function collect_entries(dir::AbstractString)
    logs = filter(f -> endswith(f, ".log"), readdir(dir))
    entries = Vector{NamedTuple}()
    for f in logs
        md = parse_dt_eps_from_filename(f)
        md === nothing && continue
        stats = parse_tail_stats(joinpath(dir, f))
        stats === nothing && continue
        push!(entries, (phi=md.phi, eps=md.eps, dt=md.dt,
                        E_pot=stats.E_pot, cc=stats.cc,
                        E_pot_err=stats.E_pot_err, cc_err=stats.cc_err))
    end
    return entries
end

function main()
    # Directory with outputs
    root = joinpath(@__DIR__, "..", "examples", "single_T_collision_calc")
    entries = collect_entries(root)
    if isempty(entries)
        @warn "No results found in" root
        return
    end
    # Unique sets
    phis = sort(unique(getfield.(entries, :phi)))
    epses = sort(unique(getfield.(entries, :eps)))

    # Aggregator across dt for given (phi, eps)
    function agg(phi, eps)
        sel = filter(e -> e.phi == phi && e.eps == eps, entries)
        if isempty(sel); return (NaN, NaN, NaN, NaN); end
        epm = getfield.(sel, :E_pot); ccm = getfield.(sel, :cc)
        epe = getfield.(sel, :E_pot_err); cce = getfield.(sel, :cc_err)
        μE = mean(epm); μC = mean(ccm)
        σE = sqrt(mean(abs2, epe) + (std(epm)^2))
        σC = sqrt(mean(abs2, cce) + (std(ccm)^2))
        return (μE, μC, σE, σC)
    end

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    fig, axs = subplots(2, 2, figsize=(14, 10))
    axE_eps = axs[1, 1]
    axC_eps = axs[1, 2]
    axE_phi = axs[2, 1]
    axC_phi = axs[2, 2]

    # E_pot and collision vs epsilon — 8 lines per phi
    for (i, phi) in enumerate(phis)
        yE = Float64[]; yC = Float64[]; yEe = Float64[]; yCe = Float64[]
        for eps in epses
            μE, μC, σE, σC = agg(phi, eps)
            push!(yE, μE); push!(yEe, max(σE, 1e-16))
            push!(yC, μC); push!(yCe, max(σC, 1e-16))
        end
        phistr = @sprintf("%.1f", phi)
        axE_eps.errorbar(epses, yE, yerr=yEe, fmt="o-", capsize=3, color=colors[(i-1)%length(colors)+1], label=L"\phi = $(phistr)")
        axC_eps.errorbar(epses, yC, yerr=yCe, fmt="s-", capsize=3, color=colors[(i-1)%length(colors)+1], label=L"\phi = $(phistr)")
    end

    # E_pot and collision vs phi — one curve per epsilon
    for (j, eps) in enumerate(epses)
        yE = Float64[]; yC = Float64[]; yEe = Float64[]; yCe = Float64[]
        for phi in phis
            μE, μC, σE, σC = agg(phi, eps)
            push!(yE, μE); push!(yEe, max(σE, 1e-16))
            push!(yC, μC); push!(yCe, max(σC, 1e-16))
        end
        epsstr = @sprintf("%.1e", eps)
        axE_phi.errorbar(phis, yE, yerr=yEe, fmt="o-", capsize=3, color=colors[(j-1)%length(colors)+1], label=L"\epsilon = $epsstr")
        axC_phi.errorbar(phis, yC, yerr=yCe, fmt="s-", capsize=3, color=colors[(j-1)%length(colors)+1], label=L"\epsilon = $epsstr")
    end

    # Formatting
    for ax in (axE_eps, axC_eps)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(true, which="both", ls=":", alpha=0.4); ax.legend(loc="best")
    end
    axE_eps.set_xlabel(L"$\epsilon$"); axE_eps.set_ylabel(L"$E_{\mathrm{pot}}$ (last-4 avg)")
    axC_eps.set_xlabel(L"$\epsilon$"); axC_eps.set_ylabel(L"collision rate (last-4 avg)")

    for ax in (axE_phi, axC_phi)
        ax.set_xscale("linear"); ax.set_yscale("log"); ax.grid(true, which="both", ls=":", alpha=0.4); ax.legend(loc="best")
    end
    axE_phi.set_xlabel(L"$\phi$"); axE_phi.set_ylabel(L"$E_{\mathrm{pot}}$ (last-4 avg)")
    axC_phi.set_xlabel(L"$\phi$"); axC_phi.set_ylabel(L"collision rate (last-4 avg)")

    tight_layout()
    outpath = joinpath(@__DIR__, "singleT_soft_sweep_epot_collisions_4plots.png")
    savefig(outpath, dpi=150)
    println("Saved figure to $(outpath)")
end

main()
