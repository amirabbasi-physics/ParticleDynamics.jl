# Accuracy analysis: partial RDFs g_OO, g_OH, g_HH and the density of
# a final frame (.gro, any precision; O, H, H per water). Intramolecular pairs are excluded.
# Threaded cell list on the CPU (julia -t auto). Writes a TOML with r, g_OO, g_OH, g_HH,
# density and the box.
# Usage: julia -t auto rdf.jl FRAME.gro OUT.toml [rmax=1.2] [dr=0.002]
using TOML, Printf, Base.Threads

function read_gro(path)
    lines = readlines(path); n = parse(Int, strip(lines[2]))
    t = lines[3][21:end]; d1 = findfirst('.', t); w = findnext('.', t, d1 + 1) - d1
    R = zeros(3, n)
    for i in 1:n
        s = lines[2 + i][21:end]
        for k in 1:3
            R[k, i] = parse(Float64, s[(k - 1) * w + 1:k * w])
        end
    end
    box = parse.(Float64, split(lines[3 + n])[1:3])
    return R, box
end

"""Histogram of distances between set A and set B (atom indices) within rmax. Pairs in
the same water are skipped; `same` counts each unordered pair once."""
function histogram(R, box, A, B, rmax, dr; same::Bool)
    nb = ceil(Int, rmax / dr)
    nc = max.(floor.(Int, box ./ rmax), 1)
    cell(x, d) = clamp(floor(Int, mod(x, box[d]) / box[d] * nc[d]), 0, nc[d] - 1)
    lin(c) = 1 + c[1] + nc[1] * (c[2] + nc[2] * c[3])
    heads = zeros(Int, prod(nc)); nxt = zeros(Int, length(B))
    for (k, j) in enumerate(B)
        c = lin((cell(R[1, j], 1), cell(R[2, j], 2), cell(R[3, j], 3)))
        nxt[k] = heads[c]; heads[c] = k
    end
    offsets(d) = nc[d] >= 3 ? (-1:1) : (0:nc[d]-1)
    hists = [zeros(Int, nb) for _ in 1:Threads.maxthreadid()]
    r2max = rmax^2
    @threads :static for ia in eachindex(A)
        h = hists[threadid()]
        i = A[ia]; xi = R[1, i]; yi = R[2, i]; zi = R[3, i]
        ci = (cell(xi, 1), cell(yi, 2), cell(zi, 3))
        for dx in offsets(1), dy in offsets(2), dz in offsets(3)
            c = (nc[1] >= 3 ? mod(ci[1] + dx, nc[1]) : dx, nc[2] >= 3 ? mod(ci[2] + dy, nc[2]) : dy,
                 nc[3] >= 3 ? mod(ci[3] + dz, nc[3]) : dz)
            k = heads[lin(c)]
            while k != 0
                j = B[k]
                if j != i && (i - 1) ÷ 3 != (j - 1) ÷ 3 && (!same || j > i)
                    ddx = R[1, j] - xi; ddy = R[2, j] - yi; ddz = R[3, j] - zi
                    ddx -= box[1] * round(ddx / box[1]); ddy -= box[2] * round(ddy / box[2])
                    ddz -= box[3] * round(ddz / box[3])
                    d2 = ddx^2 + ddy^2 + ddz^2
                    d2 < r2max && (h[1 + floor(Int, sqrt(d2) / dr)] += 1)
                end
                k = nxt[k]
            end
        end
    end
    return sum(hists)
end

function gofr(h, na, nb, V, dr; same::Bool)
    r = [(k - 0.5) * dr for k in eachindex(h)]
    pairs = same ? na * (na - 1) / 2 : na * nb
    shell = [4pi / 3 * ((k * dr)^3 - ((k - 1) * dr)^3) for k in eachindex(h)]
    return r, h ./ (pairs .* shell ./ V)
end

frame, out = ARGS[1], ARGS[2]
rmax = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 1.2
dr = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 0.002
R, box = read_gro(frame); n = size(R, 2); nw = n ÷ 3; V = prod(box)
O = collect(1:3:n); H = sort!(vcat(collect(2:3:n), collect(3:3:n)))
t = @elapsed begin
    hOO = histogram(R, box, O, O, rmax, dr; same=true)
    hOH = histogram(R, box, O, H, rmax, dr; same=false)
    hHH = histogram(R, box, H, H, rmax, dr; same=true)
end
r, gOO = gofr(hOO, length(O), length(O), V, dr; same=true)
_, gOH = gofr(hOH, length(O), length(H), V, dr; same=false)
_, gHH = gofr(hHH, length(H), length(H), V, dr; same=true)
density = nw * (15.9994 + 2 * 1.008) / 6.02214076e23 * 1e-3 / (V * 1e-27)
open(io -> TOML.print(io, Dict("frame" => frame, "n_waters" => nw, "box" => box, "density_kg_m3" => density,
    "rmax" => rmax, "dr" => dr, "r" => r, "g_OO" => gOO, "g_OH" => gOH, "g_HH" => gHH)), out, "w")
k = argmax(gOO)
@printf("%s: %d waters, density %.3f kg/m3, g_OO peak %.3f at %.4f nm (%.1f s, %d threads)\n",
        frame, nw, density, gOO[k], r[k], t, nthreads())
