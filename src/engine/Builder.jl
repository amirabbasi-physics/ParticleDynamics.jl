# Included in ParticleDynamics.DeviceEngine. Experimental cooperative 3D row construction.
# G_b=1 keeps the original serial builder; G_b=2/4/8 lanes share one atom's
# 27-cell scan (G_b=16/32 for long rows). Distance arithmetic and acceptance are identical to
# NL._kernel_neighbors3!, and a group-masked ballot compacts accepted pairs in
# lane order, so each row has the serial builder's entries in its order.

const mic_fast=NL.mic_fast

function validate_builder(lanes)
    lanes in (1,2,4,8,16,32) || throw(ArgumentError("Builder lanes must be 1, 2, 4, 8, 16 or 32"))
    return nothing
end

# Candidate kk of the cell range [·, e) for particle i1: (j, accept, tested).
@inline function _coop_test(rx,ry,rz,ids,kk,e,i1,xi::T,yi::T,zi::T,Lx::T,Ly::T,Lz::T,halfLx::T,halfLy::T,halfLz::T,rl2::T) where {T}
    j=kk<e ? (@inbounds ids[kk]) : Int32(0)
    test=(kk<e) & (j!=i1)
    accept=false
    if test
        @inbounds begin
            dx=mic_fast(rx[j]-xi,halfLx,Lx)
            dy=mic_fast(ry[j]-yi,halfLy,Ly)
            dz=mic_fast(rz[j]-zi,halfLz,Lz)
        end
        accept=muladd(dx,dx,muladd(dy,dy,dz*dz))<=rl2
    end
    return j,accept,test
end
# Group ballot of one chunk: accepted candidates to the next row slots, in lane order.
@inline function _coop_emit!(neighbors_flat,i1,N,cap,mask,below,found,j,accept)
    accepted=CUDA.vote_ballot_sync(mask,accept)
    if accept
        slot=found+(count_ones(accepted&below)%Int32)
        slot<cap && (@inbounds neighbors_flat[NL._ell_index(i1,slot,N)]=j)
    end
    return found+count_ones(accepted)%Int32
end

function _cooperative_rows3_kernel!(rx::CuDeviceVector{T},ry::CuDeviceVector{T},rz::CuDeviceVector{T},
                                    neighbors_flat,counts,cell_offsets,particle_ids_sorted,
                                    cell_of_particle,Lx::T,Ly::T,Lz::T,halfLx::T,halfLy::T,halfLz::T,
                                    nx::Int32,ny::Int32,nz::Int32,rl2::T,cap::Int32,
                                    candidates,::Val{G},::Val{S1}=Val(true)) where {T,G,S1}
    tid=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
    i1=tid÷G+1; lane=tid%G; N=length(rx)
    # A group is G consecutive lanes of one warp (G divides 32; blocks hold full
    # warps), so it exits together; ballots only name its own lanes.
    i1>N && return nothing
    warp_lane=(threadIdx().x-1)%32
    mask=G==32 ? 0xffffffff : ((UInt32(1)<<G)-UInt32(1))<<(warp_lane-lane)
    below=CUDA.lanemask(<)
    @inbounds begin
        xi=rx[i1]; yi=ry[i1]; zi=rz[i1]
        c0=cell_of_particle[i1]
        cx=c0%nx; tmp=c0÷nx; cy=tmp%ny; cz=tmp÷ny
        found=Int32(0); tested=Int64(0)
        if S1   # one cell per list radius: the 27-cell stencil, unrolled (fewer registers)
        for oz in Int32(-1):Int32(1)
            cz2=cz+oz; cz2-=(cz2>=nz)*nz; cz2+=(cz2<0)*nz
            for oy in Int32(-1):Int32(1)
                cy2=cy+oy; cy2-=(cy2>=ny)*ny; cy2+=(cy2<0)*ny
                for ox in Int32(-1):Int32(1)
                    cx2=cx+ox; cx2-=(cx2>=nx)*nx; cx2+=(cx2<0)*nx
                    c=(cz2*ny+cy2)*nx+cx2
                    s=cell_offsets[c+1]; e=cell_offsets[c+2]
                    k=s
                    while k<e # uniform within the group
                        j,accept,test=_coop_test(rx,ry,rz,particle_ids_sorted,k+Int32(lane),e,i1,xi,yi,zi,Lx,Ly,Lz,halfLx,halfLy,halfLz,rl2)
                        found=_coop_emit!(neighbors_flat,i1,N,cap,mask,below,found,j,accept)
                        if candidates!==nothing
                            tested+=count_ones(CUDA.vote_ballot_sync(mask,test))
                        end
                        k+=Int32(G)
                    end
                end
            end
        end
        else
        rl=sqrt(rl2); hx=Lx/T(nx); hy=Ly/T(ny); hz=Lz/T(nz)
        Sx=NL._stencil_reach(rl,Lx,nx); Sy=NL._stencil_reach(rl,Ly,ny); Sz=NL._stencil_reach(rl,Lz,nz)
        for oz in -Sz:Sz   # cell-row stencil (NL._row_ranges): uniform within the group
            cz2=NL._wrap_cell(cz+oz,nz); gz=NL._stencil_gap(oz,hz)
            for oy in -Sy:Sy
                cy2=NL._wrap_cell(cy+oy,ny); gy=NL._stencil_gap(oy,hy)
                g2=muladd(gy,gy,gz*gz)
                g2>rl2 && continue
                row=(cz2*ny+cy2)*nx
                rg1,rg2=NL._row_ranges(cx,g2,rl2,hx,Sx,nx)
                for seg in 1:2
                    ca,cb=seg==1 ? rg1 : rg2
                    ca>cb && continue
                    s=cell_offsets[row+ca+1]; e=cell_offsets[row+cb+2]
                    k=s
                    while k<e # uniform within the group
                        kk=k+Int32(lane)
                        accept=false; test=false; j=Int32(0)
                        if kk<e
                            j=particle_ids_sorted[kk]
                            if j!=i1
                                test=true
                                dx=mic_fast(rx[j]-xi,halfLx,Lx)
                                dy=mic_fast(ry[j]-yi,halfLy,Ly)
                                dz=mic_fast(rz[j]-zi,halfLz,Lz)
                                r2=muladd(dx,dx,muladd(dy,dy,dz*dz))
                                accept=r2<=rl2
                            end
                        end
                        accepted=CUDA.vote_ballot_sync(mask,accept)
                        if accept
                            slot=found+(count_ones(accepted&below)%Int32)
                            slot<cap && (neighbors_flat[NL._ell_index(i1,slot,N)]=j)
                        end
                        found+=count_ones(accepted)%Int32
                        if candidates!==nothing
                            tested+=count_ones(CUDA.vote_ballot_sync(mask,test))
                        end
                        k+=Int32(G)
                    end
                end
            end
        end
        end
        if lane==0
            counts[i1]=found
            candidates===nothing || (candidates[i1]=tested)
        end
    end
    return nothing
end


# Device-box wrapper: the box lengths from the geometry vector.
function _cooperative_rows3g_kernel!(rx::CuDeviceVector{T},ry,rz,neighbors_flat,counts,cell_offsets,particle_ids_sorted,
                                     cell_of_particle,geom,nx::Int32,ny::Int32,nz::Int32,rl2::T,cap::Int32,
                                     candidates,vg::Val{G},vs::Val=Val(true)) where {T,G}
    Lx,Ly,Lz=NL._boxlen(geom,Val(3))
    @inline _cooperative_rows3_kernel!(rx,ry,rz,neighbors_flat,counts,cell_offsets,particle_ids_sorted,cell_of_particle,
        Lx,Ly,Lz,T(0.5)*Lx,T(0.5)*Ly,T(0.5)*Lz,nx,ny,nz,rl2,cap,candidates,vg,vs)
    return nothing
end

const BUILDER_THREADS=128

"""Cooperative counterpart of `NL._device_finish_rebuild!` (3D). Only the row
kernel differs; capacity, commit/cause counters and reference copies reuse the
validated kernels, preserving latched-error and diagnostic semantics."""
function finish_rebuild_cooperative!(ws::NL.DeviceNeighborWorkspace{T,3},coords,refs,::Val{G}) where {T,G}
    nb=ws.list; s=ws.state; box=ws.box
    blocks=cld(Int(nb.N),256)
    halfbox=map(x->T(0.5)*x,box)
    rl2=(nb.cutoff+nb.skin)^2
    candidates=ws.diagnostics ? ws.candidates : nothing
    statistics=ws.diagnostics ? ws.statistics_partial : nothing
    @cuda threads=BUILDER_THREADS blocks=cld(Int(nb.N)*G,BUILDER_THREADS) NL._device_if_requested!(s,
        _cooperative_rows3g_kernel!,coords...,nb.neighbors_flat,nb.counts,nb.cell_offsets,
        nb.particle_ids_sorted,nb.cell_of_particle,ws.geom,nb.nx,nb.ny,nb.nz,rl2,nb.cap,
        candidates,Val(G),Val(NL._cell_sub(nb)==1))
    @cuda threads=256 blocks=blocks NL._device_if_requested!(s,NL._device_capacity_partial_kernel!,ws.capacity,nb.counts,candidates,statistics)
    @cuda threads=256 blocks=1 NL._device_commit_kernel!(s,ws.capacity,nb.cap,statistics)
    @cuda threads=256 blocks=blocks NL._device_if_requested!(s,NL._kernel_copy_refs_3d!,coords...,refs...)
    return nothing
end

