# Argon NVE/NVT/NPT comparison: shared liquid states from the Argon
# benchmark fixtures (benchmarking_packages_nve/initial_conditions/N*_density_high: ρ* 0.8,
# T* 0.85, plain LJ cutoff 2.5σ, dt* 0.0025), mapped to nm/ps/kJ mol⁻¹ (σ 0.3405 nm,
# ε 119.8 K·k_B, m 39.948) as benchmark/gromacs/Protocol.jl does, then equilibrated in GROMACS
# NVT (v-rescale τ_t 0.5 ps, 101.8 K, 20,000 steps); the final frame (positions, velocities)
# is AR_<N>.gro and its mean pressure AR_<N>.pressure (the NPT target, so the volume holds).
# Usage: julia make_states.jl FIXTURES OUTDIR N [N ...]
include(joinpath(@__DIR__, "..", "gromacs", "Protocol.jl"))
using Printf
const P = Protocol
fixtures, out = ARGS[1], ARGS[2]; mkpath(out)
const GMX = get(ENV, "GMX", "/net/storage/abbaa90/gromacs-gpu/bin/gmx")
for N in parse.(Int, ARGS[3:end])
    c = P.read_argon(joinpath(fixtures, "N$(N)_density_high"))
    name = joinpath(out, "AR_$N")
    isfile(name * ".gro") && (println("$name exists"); continue)
    s = P.SIGMA_NM; v = s / P.TAU_PS
    open(name * ".start.gro", "w") do io
        println(io, "Argon N=$N density 0.8"); println(io, N)
        for i in 1:N
            @printf(io, "%5d%-5s%5s%5d", mod(i, 100000), "AR", "AR", mod(i, 100000))
            for d in 1:3; @printf(io, "%11.6f", c.R[d, i] * s); end
            for d in 1:3; @printf(io, "%12.7f", c.V[d, i] * v); end
            println(io)
        end
        @printf(io, "%12.7f%12.7f%12.7f\n", ntuple(_ -> c.L * s, 3)...)
    end
    write(name * ".top", """
    [ defaults ]
    1 2 no 1.0 1.0
    [ atomtypes ]
    AR 18 $(P.MASS_AMU) 0.0 A $(c.sigma * s) $(c.epsilon * P.EPSILON_KJMOL)
    [ moleculetype ]
    AR 0
    [ atoms ]
    1 AR 1 AR AR 1 0.0 $(P.MASS_AMU)
    [ system ]
    Argon $N
    [ molecules ]
    AR $N
    """)
    dt = c.dt * P.TAU_PS; rc = c.rc * s; T = 0.85 * 119.8
    write(name * ".eq.mdp", """
    integrator = md
    dt = $dt
    nsteps = 20000
    cutoff-scheme = Verlet
    coulombtype = Cut-off
    rcoulomb = $rc
    vdwtype = Cut-off
    vdw-modifier = None
    rvdw = $rc
    DispCorr = no
    tcoupl = v-rescale
    tc-grps = System
    tau-t = 0.5
    ref-t = $T
    gen-vel = no
    continuation = yes
    comm-mode = None
    nstcalcenergy = 100
    nstenergy = 100
    nstxout-compressed = 0
    """)
    run(`$(joinpath(@__DIR__, "..", "water", "gro_to_g96.py")) $name.start.gro $name.start.g96`)
    run(pipeline(`$GMX grompp -f $name.eq.mdp -c $name.start.g96 -p $name.top -o $name.eq.tpr -maxwarn 3`; stdout="$name.eq.grompp.log", stderr="$name.eq.grompp.log"))
    run(pipeline(setenv(`$GMX mdrun -s $name.eq.tpr -deffnm $name.eq -nb gpu -update gpu -ntmpi 1 -ntomp 8`, merge(copy(ENV), Dict("OMP_NUM_THREADS" => "8"))); stdout="$name.eq.mdrun.log", stderr="$name.eq.mdrun.log"))
    run(pipeline(`$GMX trjconv -s $name.eq.tpr -f $name.eq.cpt -o $name.g96 -vel`; stdin=IOBuffer("0\n"), stdout=devnull, stderr=devnull))
    run(`python3 $(joinpath(@__DIR__, "..", "molten_salt", "g96_to_gro.py")) $name.g96 $name.gro $name.start.gro`)
    run(pipeline(`$GMX energy -f $name.eq.edr -o $name.pressure.xvg -b 20 -xvg none`; stdin=IOBuffer("Pressure\n"), stdout="$name.energy.log", stderr="$name.energy.log"))
    p = filter(l -> startswith(l, "Pressure"), readlines("$name.energy.log"))
    pbar = parse(Float64, split(p[1])[2]); write(name * ".pressure", string(pbar))
    @printf("AR_%d: dt %.5f ps, rc %.4f nm, T %.1f K, mean NVT pressure %.2f bar\n", N, dt, rc, T, pbar)
end
