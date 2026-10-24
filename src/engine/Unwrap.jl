# Feature: unwrapped coordinates. Every position update adds its raw
# (pre-wrap) displacement to `r*_unwrap`, as staged does; Heun midpoint moves are
# not position samples and are excluded by the kernels' call sites.

struct UnwrapFeature{U<:Tuple} <: EngineFeature
    arrays::U       # (rx_unwrap, ry_unwrap[, rz_unwrap]) of the state
end

_unwrap_arrays(st)=st.rz===nothing ? (st.rx_unwrap,st.ry_unwrap) : (st.rx_unwrap,st.ry_unwrap,st.rz_unwrap)
make_feature(::Type{UnwrapFeature},st,spec)=st.rx_unwrap===nothing ? nothing : UnwrapFeature(_unwrap_arrays(st))
feature_signature(::Type{UnwrapFeature},st,spec)=st.rx_unwrap===nothing ? nothing : map(pointer,_unwrap_arrays(st))
device_part(f::UnwrapFeature)=(unwrap=f.arrays,)
sorted_real(f::UnwrapFeature)=f.arrays
generic_integration(::UnwrapFeature)=true

"""Add the displacement `dx` of particle `i` to the unwrapped coordinates."""
@inline displace!(fx::Tuple,i,dx)=_displace!(_find(fx,Val(:unwrap)),i,dx)
@inline _displace!(::Nothing,i,dx)=nothing
@inline function _displace!(p,i,dx)
    u=p.unwrap
    for d in 1:length(u); @inbounds u[d][i]+=eltype(u[d])(dx[d]); end
    return nothing
end
