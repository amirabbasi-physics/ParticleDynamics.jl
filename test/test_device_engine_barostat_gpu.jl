module DeviceEngineBarostatGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# barostat events on the device engine. A box change
# (`set_box!`) updates the engine's device geometry in place: the next coverage
# check rebuilds the list, and the graphs (which hold the box in their kernel
# arguments) are re-instantiated lazily at the next `step!`; trial evaluations run
# on the engine without sorting (`moved`). Checks: Monte Carlo volume moves and
# stochastic cell rescaling interleaved with NVE steps match the staged golden
# records (`engine_golden.jl`: same seeds, same accept/reject decisions, same
# trajectory) while the engine is never recaptured as a whole; a box the engine's
# cell grid cannot take hands the state back to the staged path; ideal-gas MC
# samples ⟨V⟩ = (N+1)kT/P on the engine.
using Test, CUDA, ParticleDynamics, Random, Statistics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
const COMPARED=(:rx,:ry,:rz,:vx,:vy,:vz,:fx,:fy,:fz,:Epot)

# LJ liquid (ρ ≈ 0.7), unsorted (barostats require it), Float64.
function liquid(;N=500,a=1.13)
    T=Float64; pos,L=lattice(3,N,a,T); L=T(N^(1/3)*a*1.0)
    m=ceil(Int,N^(1/3)); L=T(m*a)
    st=SC.build_simulation(;N,box=(L,L,L),cutoff=T(2.5),skin=T(0.3),cap=Int32(128),neigh_interval=5,
        epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(0.002),precision=:f64,spatial_reorder=false)
    return place!(st,3,pos,velocities(3,N,1.0,T))
end
twins()=(liquid(),liquid())

"""`cycles` × (`steps` NVE steps, the last sampled, then `event!`); returns the decisions."""
function cycles!(st,event!;cycles=10,steps=25,stepper=SC.step!)
    spec=SC.nve(st); out=Bool[]
    for _ in 1:cycles
        for n in 1:steps; stepper(st,spec,st.dt;compute_energy=n==steps); end
        push!(out,event!(st))
    end
    return out
end

@testset "barostat events on the device engine" begin

@golden_testset :engine "Monte Carlo volume moves = staged golden; the engine is kept" begin
    A,B=twins()
    SC.step!(A,SC.nve(A),A.dt); e=A.engine
    @test e isa DE.StateEngine && A.tag===nothing
    mc=SC.MonteCarloVolumeMove(A;pressure=1.0,kT=1.0,log_volume_step=0.05,seed=0xB7)
    da=cycles!(A,st->SC.monte_carlo_volume_move!(st,mc))
    @test A.engine===e                                               # never recaptured as a whole
    ref=staged_reference("barostat_mc") do
        SC.run_integrator_step!(B,SC.nve(B),B.dt)
        mcb=SC.MonteCarloVolumeMove(B;pressure=1.0,kT=1.0,log_volume_step=0.05,seed=0xB7)
        db=cycles!(B,st->SC.monte_carlo_volume_move!(st,mcb);stepper=SC.run_integrator_step!)
        B,(decisions=db,box=collect(B.box3),accepts=mcb.accepts,volume_change=mcb.volume_change,
           energy_change=mcb.energy_change)
    end
    @test Float64.(da)==extra(ref,:decisions)
    @test 0<sum(da)<length(da)                                       # both outcomes exercised
    @test maximum(abs.(collect(A.box3).-extra(ref,:box)))<=1e-12*A.box3[1]
    @test isapprox(mc.volume_change,extra(ref,:volume_change);rtol=1e-10,atol=1e-10)
    @test isapprox(mc.energy_change,extra(ref,:energy_change);rtol=1e-8,atol=1e-8)
    for n in COMPARED; @test relerr(Array(getfield(A,n)),field(ref,n))<=1e-9; end
    SC.step!(A,SC.nve(A),A.dt); @test A.engine===e && e.graph_box==A.box3 && !e.stale
end

@golden_testset :engine "Stochastic cell rescaling = staged golden; the engine is kept" begin
    A,B=twins()
    SC.step!(A,SC.nve(A),A.dt); e=A.engine
    h=10*A.dt
    cr=SC.StochasticCellRescalingState(A;pressure=1.0,kT=1.0,compressibility=0.05,tau=0.1,seed=0xC5)
    cycles!(A,st->(SC.stochastic_cell_rescale!(st,cr,h); true);steps=10)
    @test A.engine===e
    ref=staged_reference("barostat_crescale") do
        SC.run_integrator_step!(B,SC.nve(B),B.dt)
        crb=SC.StochasticCellRescalingState(B;pressure=1.0,kT=1.0,compressibility=0.05,tau=0.1,seed=0xC5)
        cycles!(B,st->(SC.stochastic_cell_rescale!(st,crb,h); true);steps=10,stepper=SC.run_integrator_step!)
        B,(box=collect(B.box3),volume_change=crb.volume_change)
    end
    @test maximum(abs.(collect(A.box3).-extra(ref,:box)))<=1e-12*A.box3[1]
    @test abs(cr.volume_change)>0 && isapprox(cr.volume_change,extra(ref,:volume_change);rtol=1e-9)
    for n in COMPARED; @test relerr(Array(getfield(A,n)),field(ref,n))<=1e-9; end
end

@testset "Cell regrid in place; a box the engine's grid cannot take returns to staged" begin
    A,=(liquid(),)
    spec=SC.nve(A); for _ in 1:5; SC.step!(A,spec,A.dt); end
    e=A.engine; ws=e.run.neighbors; n0=(ws.list.nx,ws.list.ny,ws.list.nz)
    SC.set_cell_and_positions!(A,A.box3 .* 1.35)                     # more cells: storage reallocated
    @test A.engine===e && (ws.list.nx,ws.list.ny,ws.list.nz)!=n0 && e.stale_storage
    U=total_potential_energy(A)                                      # trial-style evaluation (forces invalid: staged)
    for _ in 1:5; SC.step!(A,spec,A.dt); end
    @test A.engine isa DE.StateEngine
    # A box with 3 staged cells (rc + 0.3) but fewer than 3 engine cells (rc + engine skin).
    lo=3*Float64(A.nbh.cell_size); hi=3*Float64(A.engine.run.neighbors.list.cell_size)
    @test lo<hi
    SC.set_cell_and_positions!(A,ntuple(_->(lo+hi)/2,3))
    @test A.engine===nothing                                        # handed back to the staged path
    SC.step!(A,spec,A.dt); @test A.engine isa DE.StateEngine         # recaptured with a smaller skin
end

@testset "Ideal-gas MC on the engine: ⟨V⟩ = (N+1)kT/P" begin
    T=Float64; N=216; m=6; L=T(3m)
    st=SC.build_simulation(;N,box=(L,L,L),cutoff=T(1),skin=T(0.3),cap=Int32(32),neigh_interval=20,
        nonbonded=:soft_repulsive,epsilon=T(1e-20),sigma=T(1),temperature=T(0),dt=T(0.005),precision=:f64,
        spatial_reorder=false)
    place!(st,3,[T[3*(mod((i-1)÷m^(d-1),m))-L/2+T(1.5) for i in 1:N] for d in 1:3],[zeros(T,N) for _ in 1:3])
    SC.step!(st,SC.nve(st),st.dt); e=st.engine
    P=1.0; kT=1.0
    mc=SC.MonteCarloVolumeMove(st;pressure=P,kT,log_volume_step=0.1,seed=0x16)
    for _ in 1:2000; SC.monte_carlo_volume_move!(st,mc); end
    V=Float64[]
    for _ in 1:400
        for _ in 1:10; SC.monte_carlo_volume_move!(st,mc); end
        push!(V,SC.cell_volume(st))
    end
    @test st.engine===e
    want=(N+1)*kT/P; b=reshape(V,40,10); se=std(vec(mean(b;dims=1)))/sqrt(10)   # batch means
    @test abs(mean(V)-want)<5*max(se,1e-3*want)
end
end # testset
end
