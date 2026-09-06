# ───────────────────────────────────────────────────────────────────────────────
# Bond lookup helpers for the CSR bond adjacency (not the ELL neighbor list).
# ───────────────────────────────────────────────────────────────────────────────

@inline function _is_bonded(i::Int32, j::Int32,
                            bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32})
    base = bindex[i]
    nb = bcounts[i]
    @inbounds begin
        if nb == 0
            return false
        elseif nb == 1
            return bflat[base + 1] == j
        elseif nb == 2
            b1 = bflat[base + 1]
            b2 = bflat[base + 2]
            return (b1 == j) | (b2 == j)
        elseif nb == 3
            b1 = bflat[base + 1]
            b2 = bflat[base + 2]
            b3 = bflat[base + 3]
            return (b1 == j) | (b2 == j) | (b3 == j)
        end
        for t in 0:Int(nb-1)
            if bflat[base + t + 1] == j
                return true
            end
        end
    end
    return false
end

@inline function _bond_cache(i::Int32,
                             bindex::CuDeviceVector{Int32},
                             bflat::CuDeviceVector{Int32},
                             bcounts::CuDeviceVector{Int32})
    base = bindex[i]
    nb = bcounts[i]
    b1 = Int32(0)
    b2 = Int32(0)
    @inbounds begin
        if nb >= 1
            b1 = bflat[base + 1]
        end
        if nb >= 2
            b2 = bflat[base + 2]
        end
    end
    return base, nb, b1, b2
end

@inline function _is_bonded_cached(j::Int32,
                                   base::Int32, nb::Int32, b1::Int32, b2::Int32,
                                   bflat::CuDeviceVector{Int32})
    @inbounds begin
        if nb == 0
            return false
        elseif nb == 1
            return b1 == j
        elseif nb == 2
            return (b1 == j) | (b2 == j)
        end
        for t in 0:Int(nb-1)
            if bflat[base + t + 1] == j
                return true
            end
        end
    end
    return false
end
