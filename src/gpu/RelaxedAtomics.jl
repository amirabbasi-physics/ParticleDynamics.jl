# Relaxed (monotonic) atomic accumulation for device kernels.
#
# `CUDA.@atomic a[i] += v` and `CUDA.atomic_add!` compile to sequentially consistent
# atomics: `MEMBAR.ALL.GPU; ERRBAR; ATOMG.STRONG; CCTL.IVALL` (a fence and an L1
# invalidation around every update). Accumulations whose result is only read after the
# kernel (or after a `sync_threads()` for shared memory) need atomicity, not ordering,
# which is what CUDA C's `atomicAdd` gives: a single fence-free `RED`/`ATOM` instruction.
# The additions are the same; their order was already unspecified.

for (T, lt, op) in ((Float32, "float", "fadd"), (Float64, "double", "fadd"), (Int32, "i32", "add")),
    (AS, scope) in ((1, "device"), (3, "block"))
    ir = """define void @entry(ptr addrspace($AS) %p, $lt %v) #0 {
        %old = atomicrmw $op ptr addrspace($AS) %p, $lt %v syncscope("$scope") monotonic
        ret void
    }
    attributes #0 = { alwaysinline }"""
    @eval @inline red_add!(p::Core.LLVMPtr{$T,$AS}, v::$T) =
        Base.llvmcall(($ir, "entry"), Cvoid, Tuple{Core.LLVMPtr{$T,$AS},$T}, p, v)
end

@inline _red_index(A, i) = i
@inline _red_index(A, i, j) = i + (j - 1) * size(A, 1)

"""
    @red A[i] += v
    @red A[i, j] -= v

Relaxed atomic accumulation into a device array (global or shared memory) of `Float32`,
`Float64` or `Int32`: atomic, without memory ordering. Use only where the accumulated values are read
after the kernel or after a block barrier; keep `CUDA.@atomic` or an explicit
`CUDA.threadfence()` where other threads must observe the update in order (tickets, flags).
"""
macro red(ex)
    Meta.isexpr(ex, (:+=, :-=)) && Meta.isexpr(ex.args[1], :ref) ||
        error("@red expects `A[i...] += v` or `A[i...] -= v`")
    ref = ex.args[1]; A = ref.args[1]; idx = ref.args[2:end]; v = ex.args[2]
    a = gensym(:a)
    val = ex.head === :-= ? :(-($(esc(v)))) : esc(v)
    return quote
        $a = $(esc(A))
        red_add!(pointer($a, _red_index($a, $(map(esc, idx)...))), $val)
        nothing
    end
end
