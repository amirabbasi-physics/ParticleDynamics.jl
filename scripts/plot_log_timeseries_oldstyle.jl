#!/usr/bin/env julia

# Plot a chosen quantity vs time from a VV-style .log file
# Styling matches scripts/collision_analysis_old.jl exactly (TeX + Times New Roman,
# tick params, spine widths, grid, etc.).

using PyPlot
using DelimitedFiles
using LaTeXStrings
using Printf

rc("text", usetex = true)
rc("font", family = "Times New Roman")

# Old-style global RC (match collision_analysis_old.jl)
PyPlot.rc("font", size=12)
PyPlot.rc("axes", titlesize=14)
PyPlot.rc("axes", labelsize=12)
PyPlot.rc("xtick", labelsize=10)
PyPlot.rc("ytick", labelsize=10)
PyPlot.rc("legend", fontsize=12)

function read_log_columns(path::AbstractString)
    # Return (headers::Vector{String}, cols::Vector{Vector{Float64}})
    lines = readlines(path)
    idx_header = findfirst(i -> occursin("|", lines[i]) && occursin("Time", lines[i]), eachindex(lines))
    idx_header === nothing && error("Could not find header line with 'Time' in $(path)")

    header_line = strip(lines[idx_header])
    headers = String.(strip.(split(header_line, '|')))

    # Collect numeric rows after header
    data_lines = [strip(ln) for ln in lines[idx_header+1:end] if !isempty(strip(ln))]
    ncols = length(headers)
    cols = [Float64[] for _ in 1:ncols]

    for ln in data_lines
        parts = split(ln, '|')
        length(parts) < ncols && continue
        for j in 1:ncols
            x = try
                parse(Float64, strip(parts[j]))
            catch
                NaN
            end
            push!(cols[j], x)
        end
    end
    return headers, cols
end

function find_column(headers::Vector{String}, name::AbstractString)
    # Match either exact header, or sanitized (spaces, '/', multiple spaces -> underscores, lowercased)
    function sanitize(s)
        s2 = lowercase(replace(strip(s), "/" => "_", "  " => " "))
        s2 = replace(s2, r"\s+" => "_")
        return s2
    end
    target = sanitize(name)
    for (i, h) in enumerate(headers)
        if strip(h) == name || sanitize(h) == target
            return i
        end
    end
    error("Column '$(name)' not found. Available: $(join(headers, ", "))")
end

function format_axis!(ax)
    for tick in ax.xaxis.get_major_ticks()
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_fontsize(30)
        tick.label2.set_fontsize(30)
    end
    for tick in ax.yaxis.get_major_ticks()
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_fontsize(30)
        tick.label2.set_fontsize(30)
    end
    ax[:tick_params](which="major", axis="x", direction="in", length=10, width=2.5)
    ax[:tick_params](which="minor", axis="x", direction="in", length=6, width=2.0)
    ax[:tick_params](which="major", axis="y", direction="in", length=10, width=2.5)
    ax[:tick_params](which="minor", axis="y", direction="in", length=6, width=2.0)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
end

qlabel_map(q::AbstractString) = begin
    q == "E_pot" ? L"$E_{\mathrm{pot}}$" :
    q == "E_kin" ? L"$E_{\mathrm{kin}}$" :
    q == "E_tot" ? L"$E_{\mathrm{tot}}$" :
    q == "EPR" ? L"$\mathrm{EPR}$" :
    q == "UPR" ? L"$\mathrm{UPR}$" :
    q == "cold/cold coll" ? L"$\kappa_{cc}$" :
    q == "hot/cold coll" ? L"$\kappa_{hc}$" :
    q == "hot/hot coll" ? L"$\kappa_{hh}$" :
    L"$$(q)$"
end

# Return bare LaTeX code (without $) for a known quantity name
function qlatex_code(q::AbstractString)
    q == "E_pot" && return "E_{\\mathrm{pot}}"
    q == "E_kin" && return "E_{\\mathrm{kin}}"
    q == "E_tot" && return "E_{\\mathrm{tot}}"
    q == "EPR"   && return "\\mathrm{EPR}"
    q == "UPR"   && return "\\mathrm{UPR}"
    q == "cold/cold coll" && return "\\kappa_{cc}"
    q == "hot/cold coll"  && return "\\kappa_{hc}"
    q == "hot/hot coll"   && return "\\kappa_{hh}"
    return q
end

function parse_args()
    if length(ARGS) < 1
        println("""
Usage: julia --project=. scripts/plot_log_timeseries_oldstyle.jl <logfile> [--stacked] [--linlin|--linlog|--loglog] [--skip N] [--ratio <num> <den>] [quantity1 [quantity2 ...]] [out.pdf]
Examples:
  Overlay two:   julia --project=. scripts/plot_log_timeseries_oldstyle.jl examples/traj2d_filters_vv.log E_pot E_kin out.pdf
  Stacked three: julia --project=. scripts/plot_log_timeseries_oldstyle.jl examples/traj2d_filters_vv.log --stacked "cold/cold coll" "hot/cold coll" "hot/hot coll" collisions.pdf
  Lin-Log plot:  julia --project=. scripts/plot_log_timeseries_oldstyle.jl examples/traj2d_filters_vv.log --linlog EPR "hot/cold coll"
  Log-Log plot:  julia --project=. scripts/plot_log_timeseries_oldstyle.jl examples/traj2d_filters_vv.log --loglog EPR "hot/cold coll"
  Skip first N pts: julia --project=. scripts/plot_log_timeseries_oldstyle.jl examples/traj2d_filters_vv.log --skip 20 EPR
  Ratio plot:    julia --project=. scripts/plot_log_timeseries_oldstyle.jl examples/traj2d_filters_vv.log --ratio EPR "hot/cold coll" --linlog
""")
        return nothing
    end
    logfile = ARGS[1]
    rest = collect(ARGS[2:end])
    stacked = false
    scalex = "linear"; scaley = "linear"  # default linlin
    ratio = nothing  # (numerator::String, denominator::String)
    skip_n = 0
    # detect and remove flag
    filter!.(Ref(!isempty), (rest,))
    if any(x -> x == "--stacked", rest)
        stacked = true
        rest = filter(x -> x != "--stacked", rest)
    end
    # skip option consumes an integer
    let idx = findfirst(==("--skip"), rest)
        if idx !== nothing
            if length(rest) < idx + 1
                error("--skip requires an integer argument")
            end
            skip_n = try
                parse(Int, rest[idx+1])
            catch
                error("--skip argument must be an integer; got '" * rest[idx+1] * "'")
            end
            deleteat!(rest, idx:idx+1)
        end
    end
    # ratio option consumes two following tokens as column names
    let idx = findfirst(==("--ratio"), rest)
        if idx !== nothing
            if length(rest) < idx + 2
                error("--ratio requires two names: --ratio <numerator> <denominator>")
            end
            ratio = (rest[idx+1], rest[idx+2])
            # remove the flag and its args
            deleteat!(rest, idx:idx+2)
        end
    end
    if any(x -> x == "--loglog", rest)
        scalex = "log"; scaley = "log"
        rest = filter(x -> x != "--loglog", rest)
    end
    if any(x -> x == "--linlog", rest)
        scalex = "linear"; scaley = "log"
        rest = filter(x -> x != "--linlog", rest)
    end
    if any(x -> x == "--linlin", rest)
        scalex = "linear"; scaley = "linear"
        rest = filter(x -> x != "--linlin", rest)
    end
    outpdf = nothing
    if !isempty(rest) && endswith(rest[end], ".pdf")
        outpdf = rest[end]
        pop!(rest)
    end
    quantities = isempty(rest) ? ["E_pot"] : rest
    if outpdf === nothing
        base = replace(basename(logfile), ".log" => "")
        if ratio !== nothing
            num = replace(ratio[1], ' ' => '_', '/' => '_')
            den = replace(ratio[2], ' ' => '_', '/' => '_')
            outpdf = "$(base)_$(num)_over_$(den)_timeseries.pdf"
        elseif stacked
            outpdf = "$(base)_stacked_timeseries.pdf"
        else
            joined = join(replace.(quantities, ' ' => '_', '/' => '_'), "_")
            outpdf = "$(base)_$(joined)_timeseries.pdf"
        end
    end
    return (logfile=logfile, stacked=stacked, scalex=scalex, scaley=scaley, ratio=ratio, skip_n=skip_n, quantities=quantities, outpdf=outpdf)
end

function main()
    cfg = parse_args()
    cfg === nothing && return
    logfile = cfg.logfile
    quantities = cfg.quantities
    stacked = cfg.stacked
    scalex = cfg.scalex
    scaley = cfg.scaley
    ratio = cfg.ratio
    skip_n = cfg.skip_n
    outpdf = cfg.outpdf

    headers, cols = read_log_columns(logfile)
    itime = find_column(headers, "Time")
    t = cols[itime]

    if ratio !== nothing
        num, den = ratio
        fig, ax = subplots(1, 1, figsize=(12, 8))
        inum = find_column(headers, num)
        iden = find_column(headers, den)
        yn = cols[inum]; yd = cols[iden]
        # safe division
        y = [yd[i] != 0 ? yn[i] / yd[i] : NaN for i in eachindex(yn, yd)]
        # apply skip first
        nskip = min(skip_n, length(t))
        t2 = t[nskip+1:end]; y2 = y[nskip+1:end]
        # apply scale masks
        mask = trues(length(t2))
        if scalex == "log"
            mask .= mask .& (t2 .> 0)
        end
        if scaley == "log"
            mask .= mask .& (y2 .> 0)
        end
        tt = [t2[i] for i in eachindex(t2) if mask[i]]
        yy = [y2[i] for i in eachindex(y2) if mask[i]]

        ax.plot(tt, yy, linestyle="-", linewidth=3.0, color="#194184", label="ratio")
        format_axis!(ax)
        ax.set_xlabel(L"$t$", fontsize=40)
        numc = qlatex_code(num); denc = qlatex_code(den)
        ratio_label = "\$\\frac{" * numc * "}{" * denc * "}\$"
        ax.set_ylabel(ratio_label, fontsize=40)
        ax.legend(fontsize=30, loc="best")
        ax.grid(true)
        ax.set_xscale(scalex); ax.set_yscale(scaley)
    elseif stacked && length(quantities) > 1
        fig, axes = subplots(length(quantities), 1, figsize=(12, 6 + 3 * (length(quantities)-1)))
        # axes may be a single Axes when n=1; normalize
        axlist = isa(axes, PyObject) ? [axes] : axes
        for (i, q) in enumerate(quantities)
            ival = find_column(headers, q)
            y = cols[ival]
            # apply skip first
            nskip = min(skip_n, length(t))
            t2 = t[nskip+1:end]; y2 = y[nskip+1:end]
            mask = trues(length(t2))
            if scalex == "log"
                mask .= mask .& (t2 .> 0)
            end
            if scaley == "log"
                mask .= mask .& (y2 .> 0)
            end
            tt = [t2[i] for i in eachindex(t2) if mask[i]]
            yy = [y2[i] for i in eachindex(y2) if mask[i]]
            ax = axlist[i]
            ax.plot(tt, yy, linestyle="-", linewidth=3.0, color="#194184", label=q)
            format_axis!(ax)
            ax.set_xlabel(L"$t$", fontsize=40)
            ax.set_ylabel(qlabel_map(q), fontsize=40)
            ax.legend(fontsize=30, loc="best")
            ax.grid(true)
            ax.set_xscale(scalex); ax.set_yscale(scaley)
        end
    else
        fig, ax = subplots(1, 1, figsize=(12, 8))
        colors = ["#194184", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        for (k, q) in enumerate(quantities)
            ival = find_column(headers, q)
            y = cols[ival]
            # apply skip first
            nskip = min(skip_n, length(t))
            t2 = t[nskip+1:end]; y2 = y[nskip+1:end]
            mask = trues(length(t2))
            if scalex == "log"
                mask .= mask .& (t2 .> 0)
            end
            if scaley == "log"
                mask .= mask .& (y2 .> 0)
            end
            tt = [t2[i] for i in eachindex(t2) if mask[i]]
            yy = [y2[i] for i in eachindex(y2) if mask[i]]
            ax.plot(tt, yy, linestyle="-", linewidth=3.0, color=colors[(k-1)%length(colors)+1], label=q)
        end
        format_axis!(ax)
        ax.set_xlabel(L"$t$", fontsize=40)
        if length(quantities) == 1
            ax.set_ylabel(qlabel_map(first(quantities)), fontsize=40)
        else
            ax.set_ylabel(L"value", fontsize=40)
        end
        ax.legend(fontsize=30, loc="best")
        ax.grid(true)
        ax.set_xscale(scalex); ax.set_yscale(scaley)
    end

    tight_layout()
    savefig(outpdf)
    println("Saved: $(outpdf)")
end

main()
